# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:28:15 2018

@author: rylanl
"""
from axonimaging import image_processing as ip
import os
import h5py
import numpy as np
import tifffile as tf
import scipy.stats as ss
import matplotlib.pyplot as plt

import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt

import json

curr_folder = os.path.dirname(os.path.realpath(__file__))
#check to see if JSON is being used. If so, get config from the json file
json_path=ip.full_file_path(directory=curr_folder,file_type='.json')

if len(json_path)>0:
    print ('json found, using json data for config')

    with open(json_path[0]) as json_data:
        d = json.load(json_data)
        
    #mouse_id    
    prefix=d['general']['mouse_id']

    #get the experiment folders
    expt=d['general']['expt_folders']
    
    neuropil_sub=d['roi_filtering']['neuropil_sub']
    
    #whether to extract all traces, or just those that pass a pre-filtering step
    only_filt_traces=d['roi_filtering']['only_filt_traces']
    
    data_folders=[str(os.path.join(os.path.join(os.path.dirname(curr_folder),ep),'corrected')) for ep in expt]

else:
    prefix='m392927'
    background_file_name = prefix+"Mean_All_Corrected_DS_T_projection.tif"
    #must be in correct order of acquisition!!!
    data_folders = [r"D:\2-photon data\ChAT-IRES-Cre; Ai162\181107-M392927\Auditory", 
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\181107-M392927\Spontaneous",
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\181107-M392927\Visual",
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\181107-M392927\Airpuff"]
    
    neuropil_sub=True
    only_filt_traces=True

    
save_folder = 'filtered_figures'
roi_folder='all_rois'

save_fn = prefix+'_2p_movies_all_merged.hdf5'

masks_path=ip.full_file_path(curr_folder, file_type='.hdf5', prefix='rois_and_traces')[0]
dfile = h5py.File(masks_path)

if only_filt_traces:
    print ('extracting traces ONLY from pre-filted masks, not original population of ROIs')
    num_masks=len(dfile['filt_masks'])
    good_masks=dfile['filt_masks']
    good_surr=dfile['filt_surr_masks']
    
else:
    num_masks=len(dfile['masks_center'])
    good_masks=dfile['masks_center']
    good_surr=dfile['masks_surround']

resolution = [512, 512]

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

if not os.path.isdir(roi_folder):
    os.makedirs(roi_folder)
       
file_list=[]
frame_num_tot = 0
for data_folder in data_folders:
              
    dir_files=ip.full_file_path(os.path.join(data_folder,'corrected'), prefix='corrected', exclusion='projection')         
    dir_files.sort()
    frame_num_tot+=(len(dir_files)-1)*2000+1500

    file_list.append(dir_files)

#flatten list of lists into single file
file_list=[item for items in file_list for item in items]

os.chdir(curr_folder)


data_shape = (frame_num_tot, resolution[0], resolution[1])
print ('\nWriting H5 file: ' + save_fn)
save_f = h5py.File(save_fn)
save_dset = save_f.create_dataset('2p_movie', data_shape, dtype=np.int16, compression='lzf')

start_frame = 0
data_order=[]
for curr_f in file_list:
    print ('Currently appending ' +curr_f +'\n')
    curr_mov = tf.imread(curr_f)
    end_frame = start_frame + curr_mov.shape[0]
    save_dset[start_frame : end_frame, :, :] = curr_mov
    
    #get data type
    data_order.append((os.path.dirname(curr_f),(end_frame-start_frame)))
        
    start_frame = end_frame

save_f.close()
del curr_mov

#open saved 2P movie from disk
data_f = h5py.File(save_fn, 'r')

entire_mov=np.asarray(data_f.get('2p_movie'))

traces=[]
surr_traces=[]

good_axons=[]
good_pil=[]

good_masks=[]
good_surr=[]

good_counter=0
ecc_counter=0
#extract traces
for t in range (num_masks):
    print ('\n extracting trace from mask # ' + str(t) + ' of total mask number ' + str(num_masks))
    
       
    roi= ia.WeightedROI(good_masks[t])
        
    roi_surround=ia.WeightedROI(good_masks[t])
   
    trace=roi.get_weighted_trace_pixelwise(entire_mov)
    surr_trace=roi_surround.get_weighted_trace_pixelwise(entire_mov)
    
    
    skewed=abs(ss.stats.skew(trace))
    ecc=ip.measure_eccentricity(good_masks[t])
    
    #plot and save each trace
    f = plt.figure(figsize=(20,10))
    ax1 = f.add_subplot(211)    
    ax1.imshow(ia.array_nor(tf.imread(background_file_name)), cmap='gray', clim=[0,0.075])
    
    pt.plot_mask_borders(good_masks[t], plotAxis=ax1,is_filled=True,color='green',borderWidth=1,zoom=1,alpha=0.5)
    pt.plot_mask_borders(good_surr[t], plotAxis=ax1,is_filled=True,color='red',borderWidth=1,zoom=1,alpha=0.15)
      
    plt.text(0,600,s= 'ROI skewness (absolute value) from frames = ' + str(skewed))
    plt.text(0,620,s= 'ROI eccentricity = ' + str(ecc))
    plt.text(0,640, s= 'ROI area ' + str(roi.get_binary_area()) )
      
    ax2 = f.add_subplot(212)    
            
    ax2.plot(trace, color='g')
    ax2.plot(surr_trace, color='r')
    pt.save_figure_without_borders(f, os.path.join(roi_folder, 'roi_'+str(t)+'_.png'), dpi=300)
    plt.close()  
        
    if neuropil_sub==True:
        print('subtracting neuropil. Mean value of surround/neuropil ='+ str(np.mean(surr_trace,axis=0)))
        trace=trace-surr_trace
        
       
    surr_traces.append(surr_trace)
    traces.append(trace)
print (str(ecc_counter) + ' ROIS eliminated by eccentricity filter')
print ('Filtering complete. A total of ' + str(good_counter) + ' good ROIs remain out of an original total of ' + str(num_masks)) 

print '\n plotting ...'
good_masks=np.asarray(good_masks)


f = plt.figure(figsize=(10, 10))

colors = pt.random_color(good_masks.shape[0])
bg = ia.array_nor(tf.imread(background_file_name))

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
pt.save_figure_without_borders(f_c_bg, os.path.join(save_folder, prefix+'_2P_filtered_ROIs_with_background.png'), dpi=300)
pt.save_figure_without_borders(f_c_nbg, os.path.join(save_folder, prefix+'_2P_ROIs_filtered_without_background.png'), dpi=300)
pt.save_figure_without_borders(f_s_nbg, os.path.join(save_folder, prefix+'_2P_ROI_filtered_surrounds_background.png'), dpi=300)


plt.show()

dset_n=dfile.create_dataset('axon_traces', (num_masks,np.shape(entire_mov)[0]), dtype='f')
dset_n[:,:]=np.asarray(traces)

dset_o=dfile.create_dataset('surr_traces', (num_masks,np.shape(entire_mov)[0]), dtype='f')
dset_o[:,:]=np.asarray(surr_traces)

    
dfile.close()
data_f.close()

os.remove(save_fn)




