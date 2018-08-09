import os
import logging
from shutil import copyfile

from zro import RemoteObject

class ZmqInterface(RemoteObject):
    """
    Run methods on the GUI using ZMQ.
    """
    def __init__(self, rep_port, gui_handle):
        super(ZmqInterface, self).__init__(rep_port=rep_port)
        self.gui_handle = gui_handle
        self.publishing = False

    def run_gui_method(self, method_name, **kwargs):
        """
        Runs a method in the GUI with kwargs.
        """
        getattr(self.gui_handle, method_name)(**kwargs)

    def start(self):
        self.run_gui_method("start")

    def stop(self):
        self.run_gui_method("stop")

    def load_config(self, config_path):
        self.run_gui_method("load_config", config_path=config_path)

    def save_config(self, config_path):
        self.run_gui_method("save_config", config_path=config_path)

    def set_output_path(self, path, timestamp=True):
        self.run_gui_method("set_output_path", path=path, timestamp=timestamp)

    def move_file(self, source, destination, delete_source=False):
        logging.info('Copying file:\n %s -> %s' % (source, destination))
        copyfile(source, destination)
        logging.info("... Finished!")
        if delete_source:
            os.remove(source)
            logging.info("*** Local copy removed ***")

    def copy_last_dataset(self, destination, delete_source=False):
        source = self.gui_handle.output_dir + ".h5"
        self.move_file(source, destination, delete_source)

    def run_forever(self):
        raise RuntimeError("This device is not intended to be looped.")

    def close(self):
        self._rep_sock.close()
