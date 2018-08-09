# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:34:53 2018

@author: rylanl

Motion corrects a series of movies (which have been separated into chunks) 
using the Caiman rigid motion correction.

Input: file folder of tif files and prefix that applies to all the tifs
Output: motion-corrected tif files
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import caiman as cm
import time
from caiman.motion_correction import MotionCorrect


#whether to delete mmap files or not after completion of the script
delete_mmap=False

#mouse ID or prefix
prefix='M382713'

data_folder = r"D:\2-photon data\ChAT-IRES-Cre; Ai162\180402-M377041\Aud_vis\1"
file_list = [f for f in os.listdir(data_folder) if prefix in f and f[-4:] == '.tif']
file_list.sort()
print (file_list)

os.chdir(data_folder)


# motion correction parameters
niter_rig = 1               # number of iterations for rigid motion correction
max_shifts = (15, 15)         # maximum allow rigid shift
# for parallelization split the movies in  num_splits chuncks across time
splits_rig = 56
# start a new patch for pw-rigid motion correction every x pixels
strides = (48, 48)
# overlap between pathes (size of patch strides+overlaps)
overlaps = (24, 24)
# for parallelization split the movies in  num_splits chuncks across time
splits_els = 56
upsample_factor_grid = 2    # upsample factor to avoid smearing when merging patches
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

print('\n loading movie chunks piece by piece.....\n')
movies_chained = cm.load_movie_chain(file_list)
offset_mov = np.min(movies_chained)

if 'dview' in locals():
    dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=5, single_thread=True)



mc = MotionCorrect(file_list, offset_mov,
                   dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                   splits_rig=splits_rig,
                   strides=strides, overlaps=overlaps, splits_els=splits_els,
                   upsample_factor_grid=upsample_factor_grid,
                   max_deviation_rigid=max_deviation_rigid,
                   shifts_opencv=True, nonneg_movie=True)

t0 = time.time()

mc.motion_correct_rigid(save_movie=True)
t1 = time.time()
print((t1-t0)/60)

#delete the Raw (pre-motion corrected) movie from memory
del movies_chained

#load each motion-corrected movie and save. Delete mmap file
for files in mc.fname_tot_rig:
    loaded=cm.load(files)
    name=files[:-49] +'corrected.tif'
    loaded.astype(np.int16).save(file_name=name,to32=False)

if delete_mmap==True:
    del loaded
    for files in mc.fname_tot_rig:
        os.remove(files)