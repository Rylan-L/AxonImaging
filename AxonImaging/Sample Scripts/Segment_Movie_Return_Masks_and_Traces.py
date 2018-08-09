# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:24:32 2018

@author: rylanl
"""

from axonimaging import image_segmentation as iseg
from axonimaging import image_processing as ip
from axonimaging import signal_processing as sp

import tifffile as tf
import numpy as np
import math

prefix = '04_03_2018'
mov_path=r"D:\2-photon data\ChAT-IRES-Cre; Ai162\180402-M377041\Airpuff\Motion Corrected"

frames_for_seg=2000
frame_freq=30.02 #frame sampling frequency

#check to see if movie path is a tif or a folder. If Folder, concentate the movies into one tif.
if mov_path[:-3]!='tif':
    print ('multiple tif chunks detected: concentanating in memory.....')
    mov=ip.concenate_tifs_folder(path=mov_path, save=False)

else:    
    mov=tf.imread(mov_path)
    print('Single Movie Loaded')

print ('using first ' + str(frames_for_seg) + ' frames for segmentation')
print(np.shape(mov))
mov_chunk=mov[0:frames_for_seg,:,:]

masks=iseg.svd_segment(mov=mov_chunk, std_thresh=0, downsample=2, min_size=15,num_matrices=4, gaussian_size=5,verbose=True)
    

#extract traces from masks
#----------------------------------------------------------

raw_g_traces=[]

for mask in masks:
    trace_g=ip.get_traces_from_masks(mask,mov)
    raw_g_traces.append(trace_g)

#generate neuropil masks with dilation
neuropil_masks=ip.generate_surround_masks(masks,width=6, exclusion=3)

#subtract neuropil
neuropil_traces_green=[]

for t in range (len(neuropil_masks)):
        #green channel
        trace=ip.get_traces_from_masks(roi_mask=neuropil_masks[t],movie_arr=mov)
        neuropil_traces_green.append(trace)     

neuropil_traces_green=np.array(neuropil_traces_green)

raw_g_traces=raw_g_traces-neuropil_traces_green

#filter out ROIs that have low SNR (inactive)
#Reimer:  was calculated as the log of the ratio of the peak power in the range of 0.05–0.5 Hz
#divided by the average power in the 1–3 Hz range

#calculate SNR of ROIS

green_SNR=sp.signal_to_mean_noise(traces=raw_g_traces,sample_freq=frame_freq, signal_range=[0.01,0.5], noise_range=[1,5])
print(green_SNR)

#define SNR threshold for inclusion
threshold=math.log10(10)
print(threshold)

valid_roi_pos=np.where(green_SNR>threshold)

raw_good_traces_green=[raw_g_traces[jo] for jo in range(np.shape(valid_roi_pos)[1])]

good_masks=np.array([masks[lo] for lo in range(np.shape(valid_roi_pos)[1])])

print(np.shape(good_masks))

#save the masks
np.save(prefix+'_masks.npy',good_masks)
#save the traces
np.save(prefix+'_traces.npy',raw_good_traces_green)