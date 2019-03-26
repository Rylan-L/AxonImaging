import os
import numpy as np
import h5py
import tifffile as tf

import json
from axonimaging import image_processing as ip

import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt
import scipy.ndimage as ni
import matplotlib.pyplot as plt

curr_folder = os.path.dirname(os.path.realpath(__file__))

isSave = True
is_filter = True

plot_rois=False
if plot_rois:
    plot_original_rois=False


#check to see if JSON is being used. If so, get config from the json file
json_path=ip.full_file_path(directory=curr_folder,file_type='.json')

if len(json_path)>0:
    print ('json found, using json data for configuratation of \n' + str(os.path.realpath(__file__)))

    with open(json_path[0]) as json_data:
        d = json.load(json_data)
    
    #mouse_id    
    prefix=d['general']['mouse_id']
    bg_fn =prefix+'Mean_All_Corrected_DS_T_projection.tif'
    
    #filtering params
    filter_sigma=d['post_proc']['filter_sigma']
    cut_thr=d['post_proc']['cut_thre']
    
    print ('Getting ROIs using cut_thresh of ' + str(cut_thr))
    

else:
    #if no json, params must be manually defined
    bg_fn = "M420261Mean_All_Corrected_DS_T_projection.tif"
    filter_sigma = 2. # parameters only used if filter the rois

    cut_thr = 1. # parameters only used if filter the rois

    
save_folder = 'figures'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

data_f = h5py.File('caiman_segmentation_results.hdf5')
masks = data_f['masks'].value
data_f.close()

bg = tf.imread(bg_fn)

final_roi_dict = {}

for i, mask in enumerate(masks):

    if is_filter:
        mask_nor = (mask - np.mean(mask.flatten())) / np.abs(np.std(mask.flatten()))
        mask_nor_f = ni.filters.gaussian_filter(mask_nor, filter_sigma)
        mask_bin = np.zeros(mask_nor_f.shape, dtype=np.uint8)
        mask_bin[mask_nor_f > cut_thr] = 1

    else:
        mask_bin = np.zeros(mask.shape, dtype=np.uint8)
        mask_bin[mask > 0] = 1

    mask_labeled, mask_num = ni.label(mask_bin)
    curr_mask_dict = ia.get_masks(labeled=mask_labeled, keyPrefix='caiman_mask_{:03d}'.format(i), labelLength=5)
    for roi_key, roi_mask in curr_mask_dict.items():
        final_roi_dict.update({roi_key: ia.WeightedROI(roi_mask * mask)})

print ('Total number of ROIs:' + str(len(final_roi_dict)))

if plot_rois:
    print ('plotting')
    
    if plot_original_rois:
        print ('Plotting ALL orginal rois....can be time consuming')
        f = plt.figure(figsize=(15, 8))
        ax1 = f.add_subplot(121)
        ax1.imshow(ia.array_nor(bg), vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
        colors1 = pt.random_color(masks.shape[0])
        for i, mask in enumerate(masks):
            pt.plot_mask_borders(mask, plotAxis=ax1, color=colors1[i])
        ax1.set_title('original ROIs')
        ax1.set_axis_off()
        ax2 = f.add_subplot(122)
        ax2.imshow(ia.array_nor(bg), vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
        colors2 = pt.random_color(len(final_roi_dict))
        i = 0
        for roi in final_roi_dict.values():
            pt.plot_mask_borders(roi.get_binary_mask(), plotAxis=ax2, color=colors2[i])
            i = i + 1
        ax2.set_title('filtered ROIs')
        ax2.set_axis_off()
        
    else:
        
        f = plt.figure(figsize=(15, 8))
        ax1 = f.add_subplot(121)
        ax1.imshow(ia.array_nor(bg), vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
        colors2 = pt.random_color(len(final_roi_dict))
        i = 0
        for roi in final_roi_dict.values():
            pt.plot_mask_borders(roi.get_binary_mask(), plotAxis=ax1, color=colors2[i])
            i = i + 1
        ax1.set_title('filtered ROIs')
        ax1.set_axis_off()

    if isSave:

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        f.savefig(os.path.join(save_folder, 'caiman_segmentation_filtering.pdf'), dpi=300)

#save as hdf5
cell_file = h5py.File('cells.hdf5', 'w')

i = 0
for key, value in sorted(final_roi_dict.iteritems()):
    curr_grp = cell_file.create_group('cell{:04d}'.format(i))
    curr_grp.attrs['name'] = key
    value.to_h5_group(curr_grp)
    i += 1

cell_file.close()