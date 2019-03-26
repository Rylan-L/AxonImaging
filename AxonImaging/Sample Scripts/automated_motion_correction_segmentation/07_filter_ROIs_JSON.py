# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:39:20 2019

@author: rylanl
"""
from axonimaging import image_processing as ip
import h5py
import os
import scipy.ndimage.morphology as sm
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import corticalmapping.core.PlottingTools as pt
import corticalmapping.core.ImageAnalysis as ia

import json

plt.ioff()

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

#check to see if JSON is being used. If so, get config from the json file
json_path=ip.full_file_path(directory=curr_folder,file_type='.json')

if len(json_path)>0:
    print ('json found, using json data for configuration of refining cells/axon rois')

    with open(json_path[0]) as json_data:
        d = json.load(json_data)
    #mouse_id    
    prefix=d['general']['mouse_id']
    thr = d['roi_filtering']['circ_filter_thresh']
    

else:
    prefix='M420261'
    thr = 0.20

# set ROI filter threshold, values lower are included

background_file_name = prefix+"Mean_All_Corrected_DS_T_projection.tif"


save_folder = 'filtered_figures'
raw_masks_group='masks_center'
raw_surr_group='masks_surround'
save_folder = 'filtered_figures'


avg_zproj_green=tf.imread(background_file_name)

#get the masks
masks_path=ip.full_file_path(curr_folder, file_type='.hdf5', prefix='rois_and_traces')[0]
dfile = h5py.File(masks_path)
num_masks=len(dfile['masks_center'])
masks_v2 = np.copy(dfile.get(raw_masks_group))
raw_masks=dfile.get(raw_masks_group)

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
    
    
# function to find a bounding box from a binary array
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def plot_mask_borders(mask, plotAxis=None, color='#ff0000', zoom=1, borderWidth=2,is_filled=False, **kwargs):
    """
    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """
    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    plotingMask = np.ones(mask.shape, dtype=np.uint8)
    plotingMask[np.logical_or(np.isnan(mask), mask == 0)] = 0

    if is_filled:
        currfig = plotAxis.contourf(plotingMask, levels=[0.5, 1], colors=color, **kwargs)
    else:
        currfig = plotAxis.contour(plotingMask, levels=[0.5], colors=color, linewidths=borderWidth, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plotAxis.get_ylim())
    y_lim.sort()
    plotAxis.set_ylim(y_lim[::-1])
    plotAxis.set_aspect('equal')

    return currfig

good_masks=[]
good_surr=[]
good_counter=0

for ii in range(masks_v2.shape[0]):
    mask=masks_v2[ii]
    mask[mask > 0] = 1
    temp = sm.binary_erosion(mask, iterations=2)
    rmin, rmax, cmin, cmax = bbox(mask)
    if (np.float(np.sum(mask)) / np.float((rmax - rmin)**2 + (cmax - cmin)**2)) > thr:
        masks_v2[ii] = np.nan
    elif (np.float(np.sum(mask)) / np.float((rmax - rmin)**2 + (cmax - cmin)**2)) < thr:
        
        good_masks.append(dfile[raw_masks_group][ii])  
        good_surr.append(dfile[raw_surr_group][ii]) 
            
        good_counter+=1
    
print ('\n plotting ...')
good_masks=np.asarray(good_masks)

#plot ROIS on mean projection
f = plt.figure(figsize=(20,50))
    
ax1 = f.add_subplot(931)
ax1.imshow(avg_zproj_green, cmap='gray', clim=[0,(np.amax(avg_zproj_green)/5.)])
ax1.set_axis_off()

ax2 = f.add_subplot(932)
for m in range(len(raw_masks)):
    plot_mask_borders(raw_masks[m], plotAxis=ax2, is_filled=False, color='k', borderWidth=1, zoom=1)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = f.add_subplot(933)
ax3.imshow(np.nanmean(raw_masks, axis=0), cmap='gray', clim=[0,.001])
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])


ax4 = f.add_subplot(934)
ax4.imshow(avg_zproj_green, cmap='gray', clim=[0,(np.amax(avg_zproj_green)/3.)])
for m in range(len(raw_masks)):
    plot_mask_borders(raw_masks[m], plotAxis=ax4, is_filled=False, color='r', borderWidth=1, zoom=1)
for m in range(len(masks_v2)):
    plot_mask_borders(masks_v2[m], plotAxis=ax4, is_filled=False, color='g', borderWidth=1, zoom=1)
ax4.set_axis_off()

ax8 = f.add_subplot(935)
for m in range(len(raw_masks)):
    plot_mask_borders(raw_masks[m], plotAxis=ax8, is_filled=True, color='k', borderWidth=1, zoom=1, alpha=0.1)
for m in range(len(masks_v2)):
    plot_mask_borders(masks_v2[m], plotAxis=ax8, is_filled=False, color='k', borderWidth=1, zoom=1)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.set_xticks([])
ax8.set_yticks([])

ax9 = f.add_subplot(936)
ax9.imshow(np.nanmean(masks_v2, axis=0), cmap='gray', clim=[0,.001])
ax9.set_xticklabels([])
ax9.set_yticklabels([])
ax9.set_xticks([])
ax9.set_yticks([])

pt.save_figure_without_borders(f, os.path.join(save_folder, 'Filtered_with_raw.png'), dpi=300)

colors = pt.random_color(good_masks.shape[0])
bg = ia.array_nor(avg_zproj_green)

f_c_bg = plt.figure(figsize=(10, 10))
ax_c_bg = f_c_bg.add_subplot(111)
ax_c_bg.imshow(bg, cmap='gray', vmin=0, vmax=0.25, interpolation='nearest')
f_c_nbg = plt.figure(figsize=(10, 10))
ax_c_nbg = f_c_nbg.add_subplot(111)
ax_c_nbg.imshow(np.zeros(bg.shape,dtype=np.uint8),vmin=0,vmax=1,cmap='gray',interpolation='nearest')
f_s_nbg = plt.figure(figsize=(10, 10))
ax_s_nbg = f_s_nbg.add_subplot(111)
ax_s_nbg.imshow(np.zeros(bg.shape,dtype=np.uint8),vmin=0,vmax=1,cmap='gray',interpolation='nearest')

i = 0
for mask_ind in range(good_masks.shape[0]):
    pt.plot_mask_borders(good_masks[mask_ind], plotAxis=ax_c_bg, color=colors[i], borderWidth=1)
    pt.plot_mask_borders(good_masks[mask_ind], plotAxis=ax_c_nbg, color=colors[i], borderWidth=1)
    pt.plot_mask_borders(np.asarray(good_surr)[mask_ind], plotAxis=ax_s_nbg, color=colors[i], borderWidth=1)
    i += 1

print 'saving figures ...'
pt.save_figure_without_borders(f_c_bg, os.path.join(save_folder, '2P_filtered_ROIs_with_background.png'), dpi=300)
pt.save_figure_without_borders(f_c_nbg, os.path.join(save_folder, '2P_ROIs_filtered_without_background.png'), dpi=300)
pt.save_figure_without_borders(f_s_nbg, os.path.join(save_folder, '2P_ROI_filtered_surrounds_background.png'), dpi=300)



if "filt_masks" and "filt_surr_masks" in dfile:
    print ('filtered mask datasets already exist - Overwriting')
    del dfile['filt_masks']
    del dfile['filt_surr_masks']

dset_r=dfile.create_dataset('filt_masks', (good_counter,np.shape(avg_zproj_green)[0],np.shape(avg_zproj_green)[1]), dtype='f')
dset_r[:,:,:]=np.asarray(good_masks)
    
dset_s=dfile.create_dataset('filt_surr_masks', (good_counter,np.shape(avg_zproj_green)[0],np.shape(avg_zproj_green)[1]), dtype='f')
dset_s[:,:,:]=np.asarray(good_surr)

    
dfile.close()
