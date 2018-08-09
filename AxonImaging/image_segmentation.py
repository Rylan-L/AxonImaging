# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:15:08 2018

@author: Rylan Larsen
"""

import skimage.transform as st
import skimage.morphology, skimage.segmentation
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ni
import cv2
import tifffile as tf

def svd_segment(mov, std_thresh=0, min_size=8, gaussian_size=5,num_matrices=5, downsample=2, verbose=True):
    '''Segments a movie that has been converted into an array by first denoising and reconstructing the array using
    singular value decomposition (SVD), then thresholding based on the standard deviation of the resulting denoised movie. Also includes
    size exclusion for eliminating small ROI regions.
    
    :param: mov: an array of image (movie in array form)
    :param: std_thresh: threshold in terms of std deviations for including an ROI after denoising. Default=0
                        When std_thresh is 0, the threshold is calculated as two times the minimum standard deviation]\
    :param: min_size: the minimum area (in pixels) for an ROI. Used during size exclusion.
    :param: guassian_size: guassian kernal used when filtering the ROI stdeviation image.
    :param: num_matrices: number of matrices to use when reconstructing the movie after SVD. Default is 5
    :param: downsample: amount of spatial downsampling in X and Y to do prior to finding ROIs. Used to speed calculations. Masks are converted back to the orginal mov size after segmentation.
    :param: verbose: whether to return plots of the unitary matrices, the standard deviation image, and the final ROIs on the mean time-projection image.    
    
    '''
    mean_mov_org=np.mean(mov,axis=0)
    
    if downsample!=0:
        print('Spatially downsampling by factor of: '+str(downsample) +' in x and y dimensions.')
        mov=st.downscale_local_mean(mov, (1,downsample,downsample))
    
    mean_mov=np.mean(mov,axis=0, dtype=np.float)
    
    #subtract the mean    
    mov=mov-mean_mov
    
    #SVD step
    U,s,V = np.linalg.svd(mov.reshape(mov.shape[0],mov.shape[1]*mov.shape[2]), full_matrices=0)
    
    #number of dimensions to plot
    num_dim = 5
    if verbose==True:
        f = plt.figure(figsize=(15,10))
        
        for ii in range(num_dim):
            
            ax = f.add_subplot(221)
            axis_title = 'Unitary Matrix number ' + str(ii)
            ax.set_title(axis_title)
            mask = V[ii,:].reshape(mov.shape[1],mov.shape[2])
            #plt.imshow(mask,cmap='gray',interpolation='nearest')
            
    
            ax = f.add_subplot(222)
            ax.plot(U[:,ii], color='black')
            axis_title = 'Unitary Matrix number ' + str(ii)
            ax.set_title(axis_title)
            
        
        f.savefig('SVD_segmentation_Matrices.png')
        #plt.show(ax)
    # remake with a few Unitary Matrices

    s[num_matrices:] = 0
    denoised_movie = np.dot(U * s, V).reshape(mov.shape[0],mov.shape[1],mov.shape[2]) + np.mean(mov, axis=0)
    
    ROI_image = np.std(ni.gaussian_filter1d(denoised_movie, gaussian_size, axis=0), axis=0)
    ROI_thr = np.zeros(ROI_image.shape, dtype=np.bool)
    
    if std_thresh==0:
        std_thresh=(1.15*np.mean(ROI_image))
        print(std_thresh)
        print ('Standard Deviation threshold not defined. Using 1.15 times the mean STDEV value: ' + str(std_thresh))
        
    ROI_thr[ROI_image > std_thresh] = 1
    ROI_thr = skimage.morphology.remove_small_objects(ROI_thr, min_size=min_size, connectivity=0, in_place=False)

    out = ni.distance_transform_edt(~ROI_thr)
    out = out < 0.05 * out.max()
    out = skimage.morphology.skeletonize(out)
    out = skimage.morphology.binary_dilation(out, skimage.morphology.selem.disk(1))
    out = skimage.segmentation.clear_border(out)
    out = out | ROI_thr
    
    if verbose==True:
        # plot correlation map
        f_2 = plt.figure(figsize=(20,15))
        ax = f_2.add_subplot(221)
        #plt.imshow(ROI_image, clim=(0,100))
       

        ax = f_2.add_subplot(222)
       
        
        f_2.savefig('Correlation_Map.png')
        #plt.imshow(out, cmap='Greys', alpha=0.5)
    
    # make list of binary masks
    patches, number_patches = ni.label(out)
    
    # make array with each patch in a different plane
    mask_array = np.zeros((number_patches, patches.shape[0], patches.shape[1]), dtype=bool)

    for kk in range(1, number_patches):
        a = np.copy(patches)
        a[patches == kk] = True
        a[patches != kk] = False
        mask_array[kk] = a
    del a

    # delete all planes containing no patch
    delete_array = np.zeros([0])
    
    for ii in range(1,number_patches):
        if (np.sum(mask_array[ii]) == 0):
            delete_array = np.append(delete_array,ii)
        mask_array = np.delete(mask_array, delete_array, axis=0)
    del delete_array
    
   
    print ('total number of ROIs before size exclusion = ' + str(np.shape(mask_array)[0]))
   
    #eliminate masks with total numbers of pixels less than the min amount    

    mask_list = []
    for gg in range(mask_array.shape[0]):
        if np.count_nonzero(mask_array[gg]) > min_size:
            mask_list.append(mask_array[gg])
    mask_array = np.array(mask_list)  
    
    print ('Final total number of ROIs AFTER size exclusion = ' + str(np.shape(mask_array)[0]))
    
    if downsample!=0:
        #convert masks back to 512x512
        resized_masks=[]
        for mask in mask_array:
            resized=cv2.resize(mask.astype(np.float),dsize=(int(mask.shape[1]*downsample),int(mask.shape[0]*downsample)),interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized)
        mask_array=np.array(resized_masks)
   
    if verbose==True:
        #plot ROIS on mean projection
        f_3 = plt.figure(figsize=(15,15))
    
        ax1 = f_3.add_subplot(223)
        
        ax1.set_title(axis_title)
        
        #f_3.savefig('Mean_mov_with_ROIs.png')
        
        tf.imsave('Mean_Movie_T_projection.tif',mean_mov_org.astype(np.int16))


        for mask in mask_array:
            plotting_mask = np.ones(mask.shape, dtype=np.uint8)

            plotting_mask[np.logical_or(np.isnan(mask), mask == 0)] = 0
        
            ax1.contourf(plotting_mask, levels=[0.5, 1], colors='green',alpha=0.4)
            
            # put y axis in decreasing order
            y_lim = list(ax1.get_ylim())
            y_lim.sort()
            ax1.set_ylim(y_lim[::-1])

            ax1.set_aspect('equal')
            ax1.set_axis_off()
        f_3.savefig('Final_Masks.png')
            
               
        
    #return mask_array as 512 x 512
    return mask_array


