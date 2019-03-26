# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:48:04 2018

@author: rylanl
"""
from axonimaging import image_processing as ip
import os
import h5py
import numpy as np
import tifffile as tf

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
    
    save_avi=d['general']['save_avi']
    save_down_mean=d['general']['save_down_mean']
    save_mmap=d['general']['save_mmap']
    
    save_h5=d['general']['save_down_hdf5']
    
    downsample_amount=d['general']['downsample_avi']
        

    data_folders=[str(os.path.join(os.path.join(os.path.dirname(curr_folder),ep),'corrected')) for ep in expt]

else:
    print ('JSON NOT FOUND. Using manually defined parameters' )

    prefix='M420261'
    downsample_amount=10

    

    save_avi=True
    save_down_mean=True
    save_mmap=True

    folders = [r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Auditory", 
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Spontaneous",
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Visual",
               r"D:\2-photon data\ChAT-IRES-Cre; Ai162\190115-M420261\Airpuff"]



    data_folders=[os.path.join(folder,'corrected') for folder in folders]

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    
save_fn = prefix+'_Corrected_2p_movies_all_merged_'+'DS_by_'+str(downsample_amount)+'_.hdf5'

#________End of param setting___________

#create global file list of all tiff in all folders
frame_num_tot = 0
file_list=[]
for data_folder in data_folders:
              
    dir_files=ip.full_file_path(data_folder, prefix='corrected', exclusion='projection')         

    file_list.append(dir_files)
    #calculate the anticipated number of frames to pre-allocate H5 file size
    frame_num_tot+=(len(dir_files)-1)*2000+1500

#flatten list of lists into single file
file_list=[item for items in file_list for item in items]

#sort tiffs in order they were collected to maintain temporal order
file_list.sort()
curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

down_mov=[]
#Downsample in time
for curr_f in file_list:
    print ('Downsampling: '+ curr_f)
    tif=tf.imread(curr_f)
    
    if tif.shape[0] % downsample_amount !=0:
        raise ValueError('the frame number of {} ({}) is not divisible by t_downsample_rate ({}).'
                         .format(curr_f, tif.shape[0], downsample_amount))
        
    downed=ip.downsample_time(tif,downsample_by=10,resave_as='')
    down_mov.append(downed)

down_mov=np.concatenate(down_mov,axis=0)

frame_num_tot=np.shape(down_mov)[0]
resolution=np.shape(down_mov)[1]
data_shape = (frame_num_tot, resolution, resolution)

if save_h5:
    print ('\nWriting H5 file: ' + save_fn)
    
    if os.path.isfile(save_fn):
        print ('H5 movie File EXISTS: Overwriting previous version with current')
        os.remove(save_fn)
  
    save_f = h5py.File(save_fn)
    save_dset = save_f.create_dataset('2p_movie', data_shape, dtype=np.int16, compression='lzf')
    save_dset[:, :, :] = down_mov
    
    save_f.close()

if save_avi==True:
    ip.avi_from_mov(mov=down_mov,normalize_histogram=True, output_name=prefix+'_downsampled_by_'+str(downsample_amount)+'_.avi')

if save_down_mean==True:
    tf.imsave(prefix+'Mean_All_Corrected_DS_T_projection.tif',  np.mean(down_mov, axis=0).astype(np.int16))
    
if save_mmap==True:
    add_to_mov = 10 - np.amin(down_mov)

    save_name = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'\
        .format(prefix, down_mov.shape[2], down_mov.shape[1], down_mov.shape[0])

    down_mov = down_mov.reshape((down_mov.shape[0], down_mov.shape[1] * down_mov.shape[2]), order='F').transpose()

    mov_join_mmap = np.memmap(save_name, shape=down_mov.shape, order='C', dtype=np.float32,
                             mode='w+')
    mov_join_mmap[:] = down_mov + add_to_mov
    mov_join_mmap.flush()

    if os.path.isfile('caiman_segmentation_results.hdf5'):
        print ('Caiman_Segementation File exists: Renaming previous version OLD results')
        os.rename('caiman_segmentation_results.hdf5', 'caiman_segmentation_results_OLD.hdf5')
        
        
    save_file = h5py.File(os.path.join(curr_folder, 'caiman_segmentation_results.hdf5'))
    save_file['bias_added_to_movie'] = add_to_mov
    save_file.close()
    

