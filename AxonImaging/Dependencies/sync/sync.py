#!/usr/bin/env python
"""
sync.py

Allen Instute of Brain Science

created on Oct 10 2014

@author: derricw

This module provices two types of digital input data collection that should
    produce the same output.

    1) SyncEvent is an asynchronous event-based input that registers a callback
        with the NI board and writes an event at a time to disk.

    2) SyncSampled samples the digital IO lines at a specified frequency and
        then extracts the events from the raw data after it is stopped.

Both produce HDF5 output files that can be opened in any language. The data
    will have 2 columns, with the first being the sample number, and the
    second being the IO state of that event.

Example Data:
    ________________________________
    |  sample_number  |  io_state  |
    |     123456      |      256   |
    |     234567      |      255   |
    |        ...      |      ...   |
    |______________________________|

The IO state's binary representation gives the logical state of all bits.

Consult dataset.py for an example analysis suite.

Dependencies
------------
numpy  http://www.numpy.org/
h5py   http://www.h5py.org/
qtpy
toolbox  http://stash.corp.alleninstitute.org/projects/ENG/repos/toolbox/browse

"""

import datetime
import time
import psutil
import os
import logging

import h5py as h5
import numpy as np

from toolbox.IO.nidaq import CounterInputU32, CounterInputU64,\
    DigitalInput, DigitalInputU32
from toolbox.misc.timer import timeit
from .dataset import Dataset

# we want to version the data creation separate from analysis
sync_version = 1.05

# set nice
p = psutil.Process(os.getpid())
try:
    p.nice(psutil.REALTIME_PRIORITY_CLASS)
except AttributeError as e:
    p.set_nice(psutil.REALTIME_PRIORITY_CLASS)


class Sync(object):
    """
    Samples up to 32 digital lines and saves all events with their sample
        numbers.

    Parameters
    ----------
    device : str
        NI device id.
    bits : int
        How many digital lines to sample.
    output_path : str
        Name of output file.
    freq : float
        Sample frequency.
    buffer_size : int
        Size of buffer to write to disk.
    verbose : bool
        Print more stuff out.

    Examples
    --------

    >>> ss = Sync('Dev1', 32, "output", freq=100000.0)
    >>> ss.start()
    >>> time.sleep(10)
    >>> ss.clear()

    """
    def __init__(self,
                 device,
                 bits,
                 output_path,
                 freq=100000.0,
                 buffer_size=10000,
                 verbose=False,
                 save_raw=False,
                 ):

        super(Sync, self).__init__()
        self.device = device
        self.bits = bits
        self.freq = freq
        self.buffer_size = buffer_size
        self.output_path = output_path
        self.verbose = verbose
        self.save_raw = save_raw

        self.di = DigitalInputU32(device=self.device,
                                  lines=self.bits,
                                  binary=self.output_path,
                                  clock_speed=self.freq,
                                  buffer_size=self.buffer_size,
                                  )

        self.line_labels = ["" for x in range(32)]

        self.timeouts = []

    def add_label(self, bit, name):
        """
        Adds a label for a bit.

        Parameters
        ----------
        bit : int
            Bit to label
        name : str
            Name for specified bit.

        """
        self.line_labels[bit] = name

    def start(self):
        """
        Starts the task.

        """
        self.start_time = str(datetime.datetime.now())  # get a timestamp

        self.di.start()

    def stop(self):
        """
        Stops the task.  It can be resumed.

        TODO: Should I just get rid of this?  I never use it.

        """
        self.di.stop()

    def clear(self, out_file=None):
        """
        Clears the task.  It cannot be resumed.

        Parameters
        ----------
        out_file : str
            Path to save HDF5 output.

        """
        self.di.clear()
        self.stop_time = str(datetime.datetime.now())

        self._save_hdf5(out_file)

    def _save_hdf5(self, output_file_path=None):
        """
        Saves the events from the binary file to an HDF5 file.

        Parameters
        ----------
        output_file_path : str
            Path for HDF5 file.

        """
        #save sync data
        if output_file_path:
            filename = output_file_path
        else:
            filename = self.output_path+".h5"
        data = np.fromfile(self.output_path, dtype=np.uint32)
        total_samples = len(data)

        events = self._get_events(data)

        h5_output = h5.File(filename, 'w')
        h5_output.create_dataset("data", data=events)
        #save meta data
        meta_data = self._get_meta_data()
        meta_data['total_samples'] = total_samples

        meta_data_np = np.string_(str(meta_data))
        h5_output.create_dataset("meta", data=meta_data_np)
        h5_output.close()

        #remove raw file
        if not self.save_raw:
            os.remove(self.output_path)

        if self.verbose:
            logging.info("Recorded %i events." % (len(events)-1))
            logging.info("Metadata: %s" % meta_data)
            logging.info("Saving to %s" % filename)
            try:
                ds = Dataset(filename)
                ds.stats()
                ds.close()
            except Exception as e:
                logging.warning("Failed to print quick stats: %s" % e)

    def _get_meta_data(self):
        """
        Returns a dictionary of meta_data.
        """
        from dataset import dset_version
        meta_data = {
            'ni_daq': {
                'device': self.device,
                'counter_output_freq': self.freq,
                'sample_rate': self.freq,
                'counter_bits': 32,
                'event_bits': self.bits,
            },
            'start_time': self.start_time,
            'stop_time': self.stop_time,
            'line_labels': self.line_labels,
            'timeouts': self.timeouts,
            'version': {
                'dataset': dset_version,
                'sync': sync_version,
            },
            'sampling_type': "frequency",
        }
        return meta_data

    #@timeit
    def _get_events(self, data):
        """
        Gets changes and indices of np.uint32 dataset.

        #TODO: load data into chunks in case they have a really long experiment.

        """
        initial_state = data[0]
        events = np.where(data[:-1] != data[1:])[0]

        # add 1 because we want the new value after the change
        values = data[events+1]

        #create output array
        output = np.zeros((len(events)+1, 2), dtype=np.uint32)
        output[0, 1] = initial_state
        output[1:, 0] = events
        output[1:, 1] = values

        return output



if __name__ == "__main__":

    # import signal
    # import argparse
    # import sys

    # from qtpy import QtCore

    # description = """

    # sync.py\n

    # This program creates a process that controls three NIDAQmx tasks.\n

    # 1) An event input task monitors all digital lines for rising or falling
    #     edges.\n
    # 2) A pulse generator task creates a timebase for the events.\n
    # 3) A counter counts pulses on the timebase.\n

    # """

    # parser = argparse.ArgumentParser(description=description,
    #                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("output_path", type=str, help="output data path")
    # parser.add_argument("-d", "--device", type=str,
    #                     help="NIDAQ Device to use.", default="Dev1")
    # parser.add_argument("-c", "--counter_bits", type=int, default=64,
    #                     help="Counter timebase bits.")
    # parser.add_argument("-b", "--event_bits", type=int, default=32,
    #                     help="Change detection bits.")
    # parser.add_argument("-v", "--verbose", action="store_true", default=False,
    #                     help="Print a bunch of crap.")
    # parser.add_argument("-f", "--force", action="store_true",
    #                     help="Force synchronous callbacks.")
    # parser.add_argument("-hz", "--frequency", type=float, default=10000000.0,
    #                     help="Pulse (timebase) frequency.")

    # args = parser.parse_args()

    pass