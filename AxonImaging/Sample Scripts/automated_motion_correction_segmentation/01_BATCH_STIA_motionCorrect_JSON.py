# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:47:03 2019

@author: rylanl
"""

import json
from axonimaging import image_processing as ip
import os
import time
import numpy as np

import stia.motion_correction as mc

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import corticalmapping.core.PlottingTools as pt


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = 'figures'

#output_folder= os.path.join(curr_folder, 'corrected')

#general params that should be left unchanged
align_func=mc.phase_correlation
fill_value=0.

#check to see if JSON is being used. If so, get config from the json file
json_path=ip.full_file_path(directory=curr_folder,file_type='.json')

if len(json_path)>0:
    print ('json found, using json data for config')

    with open(json_path[0], 'r') as json_data:
        d = json.load(json_data)
    
    #mouse_id    
    prefix=d['general']['mouse_id']
    
    #get the experiment folders
    expt=d['general']['expt_folders']
    
    process_num=d['motion_corr']['process_num']
    anchor_frame_ind_chunk=d['motion_corr']['anchor_frame_ind_chunk']
    anchor_frame_ind_projection=d['motion_corr']['anchor_frame_ind_projection']
    iteration_chunk=d['motion_corr']['iteration_chunk']
    iteration_projection=d['motion_corr']['iteration_projection']
    
    max_offset_chunk=(d['motion_corr']['max_offset_chunk'],d['motion_corr']['max_offset_chunk'])
    max_offset_projection=(d['motion_corr']['max_offset_projection'],d['motion_corr']['max_offset_projection'])
    
    
    data_folders=[str(os.path.join(os.path.dirname(curr_folder),ep)) for ep in expt]
  
    
#handle case when JSON not passed in    
else:
    print ('JSON NOT FOUND. Using manually defined parameters' )
    #mouse ID or prefix
    prefix=''

    process_num=6
                         
    anchor_frame_ind_chunk=8
    anchor_frame_ind_projection=0
    iteration_chunk=10
    iteration_projection=10
    max_offset_chunk=(50., 50.)
    max_offset_projection=(50., 50.)

    data_folders = [r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Auditory", 
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Spontaneous",
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Visual",
                   r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Airpuff"]
    
def run():
    #create global file list of all tiff in all folders
    file_list=[ip.full_file_path(data_folder, prefix=prefix, exclusion='corrected',file_type='.tif') for data_folder in data_folders]
    
    #flatten file list
    file_list=[item for items in file_list for item in items]
    
    #sort tiffs in order they were collected to maintain temporal order
    # file_list.sort(key=lambda x: os.path.getmtime(x))
    print (file_list)
    
    print ('Motion Correcting ' + ' ...\n')
    
    mc.align_multiple_files_iterate_anchor_multi_thread(f_paths=file_list,
                                                 output_folder=curr_folder,
                                                 process_num=process_num,
                                                 anchor_frame_ind_chunk=anchor_frame_ind_chunk,
                                                 anchor_frame_ind_projection=anchor_frame_ind_projection,
                                                 iteration_chunk=iteration_chunk,
                                                 iteration_projection=iteration_projection,
                                                 max_offset_chunk=max_offset_chunk,
                                                 max_offset_projection=max_offset_projection,
                                                 align_func=align_func,
                                                 fill_value=fill_value,
                                                 preprocessing_type=0)
    
    
        
    offsets_path = os.path.join(curr_folder, 'correction_offsets.hdf5')
    
    for data_folder in data_folders:
        print ('Currently processing folder - ' + data_folder)
        dir_files_2=ip.full_file_path(str(data_folder), prefix=prefix, exclusion='corrected',file_type='.tif')
        
        dir_files_2.sort()
       
        print('\napply paths:')
        print('\n'.join(dir_files_2))
    
        mc.apply_correction_offsets(offsets_path=offsets_path,
                                    path_pairs=zip(dir_files_2, dir_files_2),
                                    output_folder=os.path.join(data_folder, 'corrected'),
                                    process_num=process_num,
                                    fill_value=0.,
                                    avi_downsample_rate=None,
                                    is_equalizing_histogram=False)

if __name__ == "__main__":
    #measure correction time
    t0 = time.time()
    
    run()      
    
    t1 = time.time()
    total_time=((t1-t0)/60)
    #    plot the motion correction and save
        
    correction_folder=os.path.join(curr_folder, 'correction_temp')
    
    experiments=expt
    num_expts=len(experiments)
    tiff_folders=next(os.walk(os.path.dirname(curr_folder)))[1]
    tiff_folders.sort()
    tiff_folders=tiff_folders[0:num_expts]
    
    x_offsets=[]
    y_offsets=[]
    
    siz_ea_expt=[]
    
    for expts in tiff_folders:
        #iterate through each type of experiment in order. Get the HDF5 files. Open each one. Append to list.
        
    
        curr_hdfs=ip.full_file_path(correction_folder, file_type='.hdf5', prefix=prefix)
        print(curr_hdfs)
        expt_frames=0
        for correct_file in curr_hdfs:
    
            #open each motion correction HDF5
            dfile = h5py.File(correct_file)
            x=dfile['offsets'][:,0]
            y=dfile['offsets'][:,1]
    
            x_offsets.append(x)
            y_offsets.append(y)
            expt_frames+=len(x)
    
        siz_ea_expt.append(expt_frames)
        
    x_offsets=np.concatenate(x_offsets)
    y_offsets=np.concatenate(y_offsets)
    
    
    f, axarr = plt.subplots(2, sharex=True,figsize=(20,20))
    
    #axarr[0].axis('X_offsets')
    axarr[0].set_title('X motion offsets')
    axarr[0].plot(x_offsets)
    axarr[0].set_ylabel('Pixel shifts at each frame')
    axarr[0].set_ylabel('Frame number')
    
    #axarr[1].axis('Y_Offsets')
    axarr[1].set_title('Y motion offsets')
    axarr[1].plot(y_offsets)
    axarr[1].set_ylabel('Pixel shifts at each frame')
    axarr[1].set_ylabel('Frame number')
    
    
    colors=['r','g','b','orange','cyan']
    x_start=0
    for xx in range(len(siz_ea_expt)):
        #check for defined experiments which actually have no associated frames
        
        if siz_ea_expt[xx]==0:
            print ('one of experiments is lacking any motion corrected frames. Check the file naming.')
            continue
            
        axarr[0].text(x_start,np.min(y_offsets)-1,tiff_folders[xx],fontsize=16 )
        axarr[0].add_patch(patches.Rectangle((x_start,np.min(y_offsets)), siz_ea_expt[xx], 100,alpha=0.1,color=colors[xx]))
           
        axarr[1].text(x_start,np.min(y_offsets)-1,tiff_folders[xx],fontsize=16 )
        axarr[1].add_patch(patches.Rectangle((x_start,np.min(y_offsets)), siz_ea_expt[xx], 100,alpha=0.1,color=colors[xx]))
        
        x_start+=siz_ea_expt[xx]
    
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)  
        
    pt.save_figure_without_borders(f, os.path.join(save_folder, prefix+'_Motion_Correction_Shifts.png'), dpi=300)
    plt.show()
    
    print ('total time taken to correct - ' + str(total_time))