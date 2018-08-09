"""
sync_device.py

Allen Institute for Brain Science

created on 14 Dec 2015

@author: derricw

ZRO device for controlling sync.

"""
from six.moves import cPickle as pickle
from shutil import copyfile
import os
import logging
import yaml

from zro import RemoteObject

from sync import Sync


class SyncDevice(RemoteObject):
    """
 
    """
    def __init__(self, rep_port):
        super(SyncDevice, self).__init__(rep_port=rep_port)

    def init(self):
        """
        """
        self.device = 'Dev1'
        self.counter_input = 'ctr0'
        self.counter_output = 'ctr2'
        self.counter_bits = 32
        self.event_bits = 24
        self.pulse_freq = 100000.0
        self.output_path = "C:/sync/output/test.h5"
        self.line_labels = "[]"
        self.delete_on_copy = False
        logging.info("Device initialized...")

    def echo(self, data):
        """
        For testing. Just echos whatever string you send.
        """
        return data

    def start(self):
        """
        Starts an experiment.
        """
        logging.info("Starting experiment...")

        self.sync = Sync(device=self.device,
                                bits=self.event_bits,
                                output_path=self.output_path,
                                freq=self.pulse_freq,
                                buffer_size=10000,
                                verbose=True,)

        lines = eval(self.line_labels)
        for index, line in enumerate(lines):
            self.sync.add_label(index, line)

        self.sync.start()

    def stop(self, h5_path=""):
        """
        Stops an experiment and clears the NIDAQ tasks.
        """
        logging.info("Stopping experiment...")
        try:
            self.sync.stop()
        except Exception as e:
            print(e)

        self.sync.clear(h5_path)
        self.sync = None
        del self.sync

    def load_config(self, path):
        """
        Loads a configuration from a .pkl file.
        """
        logging.info("Loading configuration: %s" % path)

        with open(path, 'rb') as f:
            config = yaml.load(f)

        print(config)

        self.device = config['device']
        #self.counter_input = config['counter']
        #self.counter_output = config['pulse']
        #self.counter_bits = int(config['counter_bits'])
        self.event_bits = int(config.get('event_bits', 32))
        self.pulse_freq = float(config.get('freq', 100000.0))
        self.output_path = config['output_dir']
        self.line_labels = str(config['labels'])

    def save_config(self, path):
        """
        Saves a configuration to a .pkl file.
        """
        logging.info("Saving configuration: %s" % path)

        config = {
            'device': self.device,
            'counter': self.counter_input,
            'pulse': self.counter_output,
            'freq': self.pulse_freq,
            'output_dir': self.output_path,
            'labels': eval(self.line_labels),
            'counter_bits': self.counter_bits,
            'event_bits': self.event_bits,
        }

        with open(path, 'wb') as f:
            pickle.dump(config, f)

    def copy_arbitrary_file(self, source, destination):
        """
        Copy an arbitrary file to a specified path.

        (source, destination)

        """
        logging.info('Copying file:\n %s -> %s' % (source, destination))
        copyfile(source, destination)
        logging.info("... Finished!")
        if self.delete_on_copy:
            os.remove(source)
            logging.info("*** Local copy removed ***")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sync_device = SyncDevice(rep_port=5000)
    sync_device.run_forever()
