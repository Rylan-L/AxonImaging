'''
Created on Oct 18, 2014

@author: derricw

qtpy gui for configuring and recording sync datasets.

'''


import sys
import os
import datetime
from six.moves import cPickle as pickle
import traceback
import logging
import subprocess

import numpy as np
import h5py
from qtpy import QtCore, QtGui
import yaml

from sync_gui_layout import Ui_MainWindow
from sync.sync import Sync
from sync.dataset import Dataset

from toolbox.IO.nidaq import AnalogInput, System, Device, CounterInputU32

# set up the logger.
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
rootlogger = logging.getLogger()
rootlogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
rootlogger.addHandler(console_handler)

# some default paths
##TODO: non-windows paths
LAST_SESSION = "C:/sync/last.yml"
DEFAULT_OUTPUT = "C:/sync/output/test"
LOG_DIR = "C:/sync/logs"
CONFIG_DIR = "C:/sync/config"

# FOR REMOTE CONTROL
REPLY_PORT = 11001

# FOR "ABOUT" BOX
ABOUT_STR = """
<p>Sync GUI</p>
<p>author: derricw@alleninstitute.org</p>
<p>2015</p>
URL: <a href='http://stash.corp.alleninstitute.org/projects/ENG/repos/sync/browse'>Stash</a>
"""


class MyForm(QtGui.QMainWindow):
    """
    Simple GUI for testing the Sync program.

    Remembers state of all widgets between sessions.

    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._setup_graphics()

        self._scan_devices()

        self.analog_labels = ["" for x in range(32)]
        self.digital_labels = ["" for x in range(32)]

        self._setup_table_digital()
        self._load_state()

        self._setup_buttons()

        self._calculate_rollover()
        self._calculate_datarate()

        self.com = None
        self.com_timer = None

        self._setup_communication()

        self.running = False
        self.sync_thread = None
        self.visualizer = None

        self.write_text("Ready...")

    def _scan_devices(self):
        try:
            ni_sys = System()
            self.available_devices = ni_sys.getDevNames()
            self.ui.comboBox_device.addItems(self.available_devices)
        except Exception as e:
            logging.warning("No NI devices found.")

    def _get_device_model(self):
        """
        Gets the device model and checks to ensure compatibility.
        """
        dev_str = str(self.ui.comboBox_device.currentText())
        dev = Device(dev_str)
        model = dev.getProductType()[0]
        ai_channels = dev.getAIChannels()
        ci_channels = dev.getCIChannels()
        if len(ai_channels) > 0:
            self.ui.tabWidget.setTabEnabled(1, True)
        else:
            self.ui.tabWidget.setTabEnabled(1, False)
        self.ui.label_device_model.setText(model)
        self.ui.comboBox_ci.clear()
        if ci_channels:
            self.ui.comboBox_ci.addItems([c.split("/")[1] for c in ci_channels])
        #self._setup_table_digital()
        return dev_str, model


    def _setup_table_digital(self):
        """
        Sets up the tablewidget so that the numbering is 0:31
        """
        #set vertical labels to 0:31
        data_bits = int(self.ui.lineEdit_data_bits.text())
        labels_int = range(data_bits)
        labels_str = [str(i) for i in labels_int]

        self.ui.tableWidget_labels.setRowCount(len(labels_int))
        self.ui.tableWidget_labels.setVerticalHeaderLabels(labels_str)
        #set horizontal labels
        self.ui.tableWidget_labels.setHorizontalHeaderLabels(['DIGITAL       '])
        self.ui.tableWidget_labels.setEnabled(True)

        self._poplulate_labels(self.digital_labels)

    def _setup_table_analog(self):
        """
        Sets up the tablewidget to show analog labels
        """
        dev_str, model = self._get_device_model()
        dev = Device(dev_str)
        ai_channels = dev.getAIChannels()
        labels_int = range(len(ai_channels))
        labels_str = [str(i) for i in labels_int]

        self.ui.tableWidget_labels.setRowCount(len(ai_channels))
        self.ui.tableWidget_labels.setVerticalHeaderLabels(labels_str)
        #set horizontal labels
        self.ui.tableWidget_labels.setHorizontalHeaderLabels(['ANALOG       '])
        self.ui.tableWidget_labels.setEnabled(True)

        self._poplulate_labels(self.analog_labels)

    def _setup_table_counter(self):
        """
        Called when switching to the "counter" tab.
        """
        self.ui.tableWidget_labels.setEnabled(False)

    def _setup_buttons(self):
        """
        Setup button callbacks and icons.
        """
        self.ui.pushButton_start.clicked.connect(self._start_stop)
        self.ui.pushButton_start.setIcon(QtGui.QIcon("res/record.png"))
        self.ui.pushButton_visualize.clicked.connect(self._start_stop_visualizer)
        self.ui.lineEdit_pulse_freq.textChanged.connect(self._calculate_rollover)
        self.ui.lineEdit_analog_channels.textChanged.connect(self._calculate_datarate)
        self.ui.comboBox_dtype.currentIndexChanged.connect(self._calculate_datarate)
        self.ui.comboBox_analog_freq.currentIndexChanged.connect(self._calculate_datarate)
        self.ui.lineEdit_data_bits.textEdited.connect(self._dig_line_count_changed)
        self.ui.checkBox_ci.clicked.connect(self._ci_toggle)

        self.ui.checkBox_comm.stateChanged.connect(self._setup_communication)

        self.ui.tableWidget_labels.cellChanged.connect(self._label_changed)
        self.ui.tabWidget.currentChanged.connect(self._tab_changed)

        self.ui.comboBox_device.currentIndexChanged.connect(self._get_device_model)

        self.ui.actionQuit.triggered.connect(self.close)
        self.ui.actionLoad.triggered.connect(self.load_config)
        self.ui.actionSave.triggered.connect(self.save_config)
        self.ui.actionAbout.triggered.connect(self.about)

        # set some validators
        int_val = QtGui.QIntValidator()
        int_val.setRange(1, 32)
        self.ui.lineEdit_data_bits.setValidator(int_val)

        # float_val = QtGui.QDoubleValidator()
        # float_val.setRange(1000.0, 20000000.0)
        # self.ui.lineEdit_pulse_freq.setValidator(float_val)

    def _setup_graphics(self):
        with open("res/darkorange_stylesheet.css", "r") as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)

    def _setup_communication(self):
        """
        Attempts to set up a com interface for external control.

        ##TRIGGER WARNING: this is ugly fix it
        """
        state = self.ui.checkBox_comm.checkState()
        if state:
            try:
                sys.path.append("..")
                from zmq_interface.gui_interface import ZmqInterface
            except ImportError as e:
                self.write_text("ZMQ interface failed to import.  No remote control for this session.")
                self.disable_visualizer()
                return
            try:
                ##TODO: let user specify ports
                self.com = ZmqInterface(rep_port=REPLY_PORT,
                                        gui_handle=self)
            except Exception as e:
                #traceback.print_exc(file=sys.stdout)
                self.write_text("ZMQ interface failed to start.  No remote control for this session. Reason: %s" % e)
                self.disable_visualizer()
                return
            self.start = self._start_session
            self.stop = self._stop_session
            self.load_config = self._load_state
            self.save_config = self._save_state
            self.com_timer = QtCore.QTimer()
            self.com_timer.timeout.connect(self._check_coms)
            self.com_timer.start(200)
            self.write_text("ZMQ interface set up.  Reply port on %s" % self.com.rep_port)
            self.enable_visualizer()
        else:
            if self.com:
                self.com.close()
            if self.com_timer:
                self.com_timer.stop()
            self.com = None
            self.com_timer = None
            self.enable_visualizer()
            self.write_text("ZMQ interface closed.")


    def _check_coms(self):
        """
        Check data on the comm interface.
        """
        self.com._check_rep()

    def write_text(self, text):
        """
        Write text to the output box.
        """
        self.ui.plainTextEdit.appendPlainText(text)
        logging.info(text)
        
    def set_output_path(self, path, timestamp=True):
        """
        Sets the output path text field.  This determines that output file
            directory once the experiment is started.
        """
        self.ui.lineEdit_output_path.setText(path)
        self.ui.checkBox_timestamp.setChecked(timestamp)

    def load_config(self, path=""):
        """
        Loads a specified config file.  If no file specified, will select using
            file dialog.
        """
        if not path:
            if not os.path.isdir(CONFIG_DIR):
                os.makedirs(CONFIG_DIR)
            file_path = QtGui.QFileDialog.getOpenFileName(self,
                                                          "Open Config",
                                                          CONFIG_DIR,
                                                          "Config Files (*.yml, *.yaml)")
            if not file_path:
                return
        else:
            file_path = path
        self._load_state(file_path)
        #self.write_text("Loaded config @ {}".format(file_path))

    def save_config(self, path=""):
        """
        Saves the config file to the path specified.  If none specified, will
            select the save path using a file dialog.
        """
        if not path:
            if not os.path.isdir(CONFIG_DIR):
                os.makedirs(CONFIG_DIR)
            file_path = QtGui.QFileDialog.getSaveFileName(self,
                                                          "Save Config",
                                                          CONFIG_DIR,
                                                          "Config File (*.yml, *.yaml))")
            if not file_path:
                return
        else:
            file_path = path
        self._save_state(file_path)
        self.write_text("Saved config @ {}".format(file_path))

    def _start_stop(self):
        """
        Callback for start/stop button press.
        """
        if not self.running:
            #get configuration from gui
            self._start_session()
        else:
            self._stop_session()

    def _disable_ui(self):
        """
        Disables the ui.
        """
        self.ui.tableWidget_labels.setEnabled(False)
        self.ui.groupBox.setEnabled(False)
        self.ui.groupBox_analog.setEnabled(False)
        self.ui.pushButton_visualize.setEnabled(False)

    def _enable_ui(self):
        """
        Enables the UI.
        """
        self.ui.tableWidget_labels.setEnabled(True)
        self.ui.groupBox.setEnabled(True)
        self.ui.groupBox_analog.setEnabled(True)
        self.ui.pushButton_visualize.setEnabled(True)

    def _start_session(self, path=""):
        """
        Starts a session.
        """
        if self.running:
            raise RuntimeError("Must stop session before starting.")
        now = datetime.datetime.now()
        if not path:
            self.output_dir = str(self.ui.lineEdit_output_path.text())
        else:
            self.output_dir = path
            self.ui.lineEdit_output_path.setText(path)
        if self.output_dir[-3:] == ".h5":
            self.output_dir = self.output_dir[:-3]
        if self.ui.checkBox_timestamp.isChecked():
            self.output_dir += now.strftime('%y%m%d%H%M%S')
        basedir = os.path.dirname(self.output_dir)
        try:
            os.makedirs(basedir)
        except:
            pass
        device = str(self.ui.comboBox_device.currentText())

        data_bits = int(self.ui.lineEdit_data_bits.text())
        freq = float(str(self.ui.lineEdit_pulse_freq.text()))

        #add in analog recording?
        analog_on = self.ui.checkBox_analog_channels.isChecked()
        analog_channels = eval(str(self.ui.lineEdit_analog_channels.text()))
        analog_sample_rate = float(str(self.ui.comboBox_analog_freq.currentText()))
        analog_dtype = int(str(self.ui.comboBox_dtype.currentText()))

        #add in counter input?
        counter_input_on = self.ui.checkBox_ci.isChecked()
        counter_input_terminal = str(self.ui.comboBox_ci.currentText())

        # #create Sync object
        params = {
            'device': device,
            'output_dir': self.output_dir,
            'event_bits': data_bits,
            'freq': freq,
            'labels': self.digital_labels,
            'analog_on': analog_on,
            'analog_channels': analog_channels,
            'analog_sample_rate': analog_sample_rate,
            'analog_dtype': analog_dtype,
            'analog_labels': self.analog_labels,
            'counter_input_on': counter_input_on,
            'counter_input_terminal': counter_input_terminal,
            }

        self.sync = SyncObject(params=params)
        if self.sync_thread:
            self.sync_thread.terminate()
        self.sync_thread = QtCore.QThread()
        self.sync.moveToThread(self.sync_thread)
        self.sync_thread.start()
        self.sync_thread.setPriority(QtCore.QThread.TimeCriticalPriority)

        QtCore.QTimer.singleShot(100, self.sync.start)

        self.write_text("***Starting session at \
            %s on %s ***" % (str(now), device))
        
        self.running = True
        self._disable_ui()

        self._set_visualizer_path(self.output_dir)

        self.ui.pushButton_start.setIcon(QtGui.QIcon("res/stop.png"))

    def _stop_session(self):
        """
        Ends the session.
        """
        if not self.running:
            raise RuntimeError("No session running.")

        self._release_visualizer_data()

        now = datetime.datetime.now()
        #self.sync.clear()
        QtCore.QTimer.singleShot(100, self.sync.clear)
        #self.sync = None

        self.write_text("***Ending session at \
            %s ***" % str(now))

        self.running = False
        self._enable_ui()
        self.ui.pushButton_start.setIcon(QtGui.QIcon("res/record.png"))

    def enable_visualizer(self):
        self.ui.pushButton_visualize.setEnabled(True)
        logging.info("Visualizer enabled.")

    def disable_visualizer(self):
        self.ui.pushButton_visualize.setEnabled(False)
        logging.info("Visualizer disabled.")

    def _start_visualizer(self):
        self.visualizer_process = subprocess.Popen(["python", 'visualizer.py'])
        self.visualizer_process.poll()

        from zro import DeviceProxy
        
        self.visualizer = DeviceProxy('localhost', 12000, timeout=10.0)

        logging.info("Visualizer started on port {}".format(12000))

    def _stop_visualizer(self):
        try:
            self.visualizer.close()
        except Exception as e:
            logging.warning("Visualizer communication timed out.  Is it already closed?")
        self.visualizer_process.wait()
        self.visualizer_process = None
        self.visualizer = None
        logging.info("Visualizer stopped.")

    def _start_stop_visualizer(self):
        if self.visualizer:
            self._stop_visualizer()
        else:
            self._start_visualizer()

    def _set_visualizer_path(self, path):
        if self.visualizer:
            try:
                self.visualizer.set_data_file(path)
            except Exception as e:
                logging.warning("Failed to set visualizer data file: {}".format(e))

    def _release_visualizer_data(self):
        if self.visualizer:
            try:
                self.visualizer.release_file()
            except Exception as e:
                logging.warning("Failed to release visualizer data: {}".format(e))

    def _label_changed(self, row, column):
        """
        Callback for a label change in the tablewidget.
        """
        new_value = str(self.ui.tableWidget_labels.item(row, column).text())
        tab = self.ui.tabWidget.currentIndex()
        if tab == 0:
            # digital
            self.digital_labels[row] = new_value
        elif tab == 1:
            # analog
            self.analog_labels[row] = new_value


    def _dig_line_count_changed(self, text):
        """
        Callback for user changing the digital line count.
        """
        self._setup_table_digital()

    def _ci_toggle(self, value):
        """
        Callback for counter input checkbox check state changed.
        """
        self.ui.comboBox_ci.setEnabled(value)

    def _tab_changed(self, index):
        """
        The tab (Digital/Analog/Counter) changed.
        """
        if index == 0:
            #self._poplulate_labels(self.digital_labels)
            self._setup_table_digital()
        elif index == 1:
            #self._poplulate_labels(self.analog_labels)
            self._setup_table_analog()
        else:
            self._setup_table_counter()


    def _save_state(self, config_path=""):
        """
        Saves widget states to default path.  Optionally use a custom path.
        """
        state = {
            'output_dir': str(self.ui.lineEdit_output_path.text()),
            'device': str(self.ui.comboBox_device.currentText()),
            'event_bits': str(self.ui.lineEdit_data_bits.text()),
            'freq': str(self.ui.lineEdit_pulse_freq.text()),
            'labels': self.digital_labels,
            'timestamp': self.ui.checkBox_timestamp.isChecked(),
            'comm_interface': self.ui.checkBox_comm.isChecked(),
            'analog_on': self.ui.checkBox_analog_channels.isChecked(),
            'analog_channels': eval(str(self.ui.lineEdit_analog_channels.text())),
            'analog_sample_rate': self.ui.comboBox_analog_freq.currentIndex(),
            'analog_dtype': self.ui.comboBox_dtype.currentIndex(),
            'analog_labels': self.analog_labels,
            'counter_input_terminal': str(self.ui.comboBox_ci.currentText()),
            'counter_input_on': self.ui.checkBox_ci.isChecked(),
        }
        if not config_path:
            config_path = LAST_SESSION
        with open(config_path, 'wb') as f:
            yaml.dump(state, f)

    def _load_state(self, config_path=""):
        """
        Loads previous widget states.  Optionally, load some arbirtary config.
        """
        if not config_path:
            config_path = LAST_SESSION
        try:
            with open(config_path, 'r') as f:
                data = yaml.load(f)
            self.ui.lineEdit_output_path.setText(data['output_dir'])

            #self.ui.lineEdit_device.setText(data['device'])
            dev = data['device']
            index = self.ui.comboBox_device.findText(dev)
            if index != -1:
                self.ui.comboBox_device.setCurrentIndex(index)
                self._get_device_model()

            self.ui.lineEdit_data_bits.setText(data['event_bits'])
            self.ui.lineEdit_pulse_freq.setText(data['freq'])
            self.ui.checkBox_timestamp.setChecked(data['timestamp'])
            self.ui.checkBox_comm.setChecked(data['comm_interface'])
            self.ui.checkBox_analog_channels.setChecked(data['analog_on'])
            self.ui.lineEdit_analog_channels.setText(str(data['analog_channels']))
            self.ui.comboBox_analog_freq.setCurrentIndex(data['analog_sample_rate'])
            self.ui.comboBox_dtype.setCurrentIndex(data['analog_dtype'])

            self.digital_labels = data['labels']
            self.analog_labels = data.get("analog_labels", self.analog_labels)

            # counter input
            index = self.ui.comboBox_ci.findText(data.get("counter_input_terminal", "ctr0"))
            if index != -1:
                self.ui.comboBox_ci.setCurrentIndex(index)
            self.ui.checkBox_ci.setChecked(data.get("counter_input_on", False))
            self.ui.comboBox_ci.setEnabled(self.ui.checkBox_ci.isChecked())

            # go ahead and polulate the table with the digital labels
            self.ui.tabWidget.setCurrentIndex(0)
            self._setup_table_digital()
            
            self.write_text("Loaded config @ {} successfully.".format(config_path))
        except Exception as e:
            print(e)
            self.write_text("Couldn't load previous session.  Using defaults.")

    def _poplulate_labels(self, label_list):
        for index, label in enumerate(label_list):
                self.ui.tableWidget_labels.setItem(index, 0,
                    QtGui.QTableWidgetItem(label))

    def _calculate_rollover(self):
        """
        Calculates the rollover time for the current freqency.
        """
        freq = float(str(self.ui.lineEdit_pulse_freq.text()))
        bits_per_sec = freq*32
        bytes_per_sec = bits_per_sec/8
        bytes_per_hour = bytes_per_sec*3600
        mb_per_hour = bytes_per_hour/1000000
        timestr = "%s MB/hr" % mb_per_hour
        self.ui.label_rollover.setText(timestr)

    def _calculate_datarate(self):
        try:
            channels = eval(str(self.ui.lineEdit_analog_channels.text()))
            if isinstance(channels, (list, tuple)):
                no_channels = len(channels)
            else:
                raise Exception("can't parse channels")
        except Exception as e:
            self.ui.label_datarate.setText("??? MB/hr")
            return
        databits = int(str(self.ui.comboBox_dtype.currentText()))
        sample_rate = float(str(self.ui.comboBox_analog_freq.currentText()))
        bytes_per_sec = no_channels*databits*sample_rate/8  # bytes/sec
        bytes_per_hour = bytes_per_sec*3600  # bytes/hr
        mb_per_hour = bytes_per_hour/1000000
        timestr = "{} MB/hr".format(mb_per_hour)
        self.ui.label_datarate.setText(timestr)

    def about(self):
        from sync import __version__
        mb = QtGui.QMessageBox(self)
        mb.setTextFormat(QtCore.Qt.RichText)
        mb.setText(ABOUT_STR)
        mb.setWindowTitle("Sync Gui version %s" % __version__)
        mb.show()

    def closeEvent(self, event):
        if self.running:
            logging.warning("Can't close while experiment is running.")
            event.ignore()
        else:
            logging.info("Closing...")
            if self.visualizer:
                self._stop_visualizer()
            self._save_state()
            if self.sync_thread:
                self.sync_thread.terminate()


class SyncObject(QtCore.QObject):
    """
        Thread for controlling sync.

        ##TODO: Fix params argument to not be stupid.
    """

    def __init__(self, parent=None, params={}):

        QtCore.QObject.__init__(self, parent)

        self.params = params

        self.ai = None
        self.ci = None
        self.ai_output_path = ""
        self.di_output_path = ""


    def start(self):
        #create Sync object
        self.di_output_path = self.params['output_dir']

        self.sync = Sync(device=self.params['device'],
                         bits=self.params['event_bits'],
                         output_path=self.params['output_dir'],
                         freq=self.params['freq'],
                         buffer_size=int(self.params['freq']/10),  # buffer size 1/10 the freq
                         verbose=True,)

        for i, label in enumerate(self.params['labels']):
            self.sync.add_label(i, label)

        # do we want to take a supplementary analog signal as well?
        # THIS FUNCTIONALITY SHOULD PROBABLY BE MOVED TO SYNC.PY
        # I'll have to think on that.
        if self.params['analog_on']:
            #trigger = self.sync.di.get_start_trigger_term()

            self.ai_output_path = self.params['output_dir']+"-analog"
            device = self.params['device']
            channels = self.params['analog_channels']
            dtype = eval("np.float{}".format(self.params['analog_dtype']))

            self.ai = AnalogInput(device=device,
                                  channels=channels,
                                  clock_speed=self.params['analog_sample_rate'],
                                  binary=self.ai_output_path,
                                  buffer_size=1000,
                                  dtype=dtype)
            trigger = self.sync.di.get_start_trigger_term()
            self.ai.cfg_dig_start_trigger(source=trigger,
                                          edge="rising")

            self.ai.start()

        # we'd also like to record a counter input
        if self.params['counter_input_on']:
            self.ci_output_path = self.params['output_dir']+"-counter"
            device = self.params['device']
            counter = self.params['counter_input_terminal']

            self.ci = CounterInputU32(device=device,
                                      counter=counter)
            self.ci.setup_file_output(self.ci_output_path,
                                      buffer_size=1000)
            self.ci.cfg_sample_clock(self.params['freq'],
                                     source="di/SampleClock")
            #self.ci.setCountEdgesTerminal("di/SampleClock")
            # trigger = self.sync.di.get_start_trigger_term()
            # self.ci.cfg_dig_start_trigger(source=trigger,
            #                               edge='rising')
            self.ci.start()


        self.sync.start()

    def clear(self):
        self.sync.clear()

        if self.ai:
            self.ai.clear()

            size = self.ai.buffer_size
            count = self.ai.buffercount
            samples = size*count
            channels = self.params['analog_channels']
            dtype = eval("np.float{}".format(self.params['analog_dtype']))
            sample_rate = self.params['analog_sample_rate']

            labels = self.params['analog_labels']

            logging.info("Total analog samples written per channel: {}".format(samples))
            logging.info("Adding analog data to h5 file...")
            
            datafile = h5py.File(self.di_output_path+".h5", 'a')
            analog_metadata = {
                "analog_sample_rate": sample_rate,
                "analog_channels": channels,
                "analog_samples_acquired": samples,
                "analog_dtype": np.dtype(dtype).name,
                "analog_labels": labels,
            }
            logging.info("Analog metadata: {}".format(analog_metadata))
            analog_metadata_np = np.string_(str(analog_metadata))
            datafile.create_dataset("analog_meta", data=analog_metadata_np)
            # Add analog data to hdf5 file.
            try:
                from toolbox.misc.hdf5tools import chunk2hdf5
                try:
                    result = chunk2hdf5(h5_file=datafile,
                                        data_file=self.ai_output_path,
                                        dtype=dtype,
                                        data_shape=(-1,len(channels)),
                                        data_name="analog_data",)
                    # check to ensure that the correct amount of samples
                    #   transferred.
                    if result == samples*len(channels):
                        logging.info("Data transferred, removing binary file.")
                        os.remove(self.ai_output_path)
                    else:
                        logging.warning("Data transfer questionable, leaving binary file intact.")
                except Exception as e:
                    logging.exception("Failed to add analog dataset to h5.")
            except ImportError as e:
                logging.exception("Failed to import chunk2hdf5.  Update toolbox.")

            #datafile.close()
        if self.ci:
            # clear the task
            self.ci.clear()

            # load the binary data
            counter_data = np.fromfile(self.ci_output_path, dtype=np.uint32)
            #print counter_data

            # open the hdf5 file
            datafile = h5py.File(self.di_output_path+".h5", 'a')

            # extract the event timepoints
            samples = datafile['data'].value[:, 0]
            counter_events = counter_data[samples]

            # add values to data file
            datafile.create_dataset("counter_data", data=counter_events)

            # add some metadata
            counter_metadata = {
                "counter_input_terminal": self.params['counter_input_terminal']
            }
            logging.info("Counter metadata: {}".format(counter_metadata))
            counter_metadata_np = np.string_(str(counter_metadata))
            datafile.create_dataset("counter_meta", data=counter_metadata_np)

            try:
                os.remove(self.ci_output_path)
            except Exception as e:
                logging.warning("Failed to clean up counter binary file.")

            #datafile.close()



if __name__ == "__main__":

    #add a log file handler to logging
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    dtstring = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    log_file_name = os.path.join(LOG_DIR, dtstring)+".log"
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(log_formatter)
    rootlogger.addHandler(file_handler)

    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
