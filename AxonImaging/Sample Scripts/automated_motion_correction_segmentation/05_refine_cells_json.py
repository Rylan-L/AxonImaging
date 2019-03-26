# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:44:42 2015

@author: junz
"""
import os
import h5py
import numpy as np
import operator
import matplotlib.pyplot as plt
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.FileTools as ft
import corticalmapping.core.PlottingTools as pt

import json
from axonimaging import image_processing as ip


plt.ioff()

curr_folder = os.path.dirname(os.path.realpath(__file__))

#check to see if JSON is being used. If so, get config from the json file
json_path=ip.full_file_path(directory=curr_folder,file_type='.json')

if len(json_path)>0:
    print ('json found, using json data for configuration of refining cells/axon rois')

    with open(json_path[0]) as json_data:
        d = json.load(json_data)
    
    #mouse_id    
    prefix=d['general']['mouse_id']
    bg_fn =prefix+'Mean_All_Corrected_DS_T_projection.tif'
    
    #filtering params
    center_margin = d['post_proc']['center_margin']
    area_range = d['post_proc']['area_range']
    
    overlap_thr = d['post_proc']['overlap_thresh']
  
else:
    #if no json, params must be manually defined
    bg_fn = "M420261Mean_All_Corrected_DS_T_projection.tif"
  
    # pixels, masks with center location within this pixel region at the image border will be discarded
    center_margin = [25, 25]

    # area range, range of number of pixels of a valid roi
    area_range = [45,4500]

    # for the two masks that are overlapping, if the ratio between overlap and the area of the smaller mask is larger than
    #this value, the smaller mask will be discarded.
    overlap_thr = 0.30

save_folder = 'figures'

data_file_name = 'cells.hdf5'
save_file_name = 'cells_refined.hdf5'


os.chdir(curr_folder)

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

# read cells
dfile = h5py.File(data_file_name)
cells = {}
for cellname in dfile.iterkeys():
    cells.update({cellname:ia.WeightedROI.from_h5_group(dfile[cellname])})

print 'total number of cells:', len(cells)

# get the names of cells which are on the edge
edge_cells = []
for cellname, cellmask in cells.iteritems():
    dimension = cellmask.dimension
    center = cellmask.get_center()
    if center[0] < center_margin[0] or \
       center[0] > dimension[0] - center_margin[0] or \
       center[1] < center_margin[1] or \
       center[1] > dimension[1] - center_margin[1]:

        # cellmask.plot_binary_mask_border(color='#ff0000', borderWidth=1)
        # plt.title(cellname)
        # plt.show()

        edge_cells.append(cellname)

print '\ncells to be removed because they are on the edges:'
print '\n'.join(edge_cells)

# remove edge cells
for edge_cell in edge_cells:
    _ = cells.pop(edge_cell)

# get dictionary of cell areas
cell_areas = {}
for cellname, cellmask in cells.iteritems():
    cell_areas.update({cellname: cellmask.get_binary_area()})


# remove cellnames that have area outside of the area_range
invalid_cell_ns = []
for cellname, cellarea in cell_areas.items():
    if cellarea < area_range[0] or cellarea > area_range[1]:
        invalid_cell_ns.append(cellname)
print "cells to be removed because they do not meet area criterion:"
print "\n".join(invalid_cell_ns)
for invalid_cell_n in invalid_cell_ns:
    cell_areas.pop(invalid_cell_n)


# sort cells with their binary area
cell_areas_sorted = sorted(cell_areas.items(), key=operator.itemgetter(1))
cell_areas_sorted.reverse()
cell_names_sorted = [c[0] for c in cell_areas_sorted]
# print '\n'.join([str(c) for c in cell_areas_sorted])

# get the name of cells that needs to be removed because of overlapping
retain_cells = []
remove_cells = []
for cell1_name in cell_names_sorted:
    cell1_mask = cells[cell1_name]
    is_remove = 0
    cell1_area = cell1_mask.get_binary_area()
    for cell2_name in retain_cells:
        cell2_mask = cells[cell2_name]
        cell2_area = cell2_mask.get_binary_area()
        curr_overlap = cell1_mask.binary_overlap(cell2_mask)

        if float(curr_overlap) / cell1_area > overlap_thr:
            remove_cells.append(cell1_name)
            is_remove = 1
            print cell1_name, ':', cell1_mask.get_binary_area(), ': removed'

            # f = plt.figure(figsize=(10,10))
            # ax = f.add_subplot(111)
            # cell1_mask.plot_binary_mask_border(plotAxis=ax, color='#ff0000', borderWidth=1)
            # cell2_mask.plot_binary_mask_border(plotAxis=ax, color='#0000ff', borderWidth=1)
            # ax.set_title('red:'+cell1_name+'; blue:'+cell2_name)
            # plt.show()
            break

    if is_remove == 0:
        retain_cells.append(cell1_name)
        print cell1_name, ':', cell1_mask.get_binary_area(), ': retained'

print '\ncells to be removed because of overlapping:'
print '\n'.join(remove_cells)

print '\ntotal number of reatined cells:', len(retain_cells)

# plotting
colors = pt.random_color(len(cells.keys()))
bgImg = tf.imread(bg_fn)

f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.imshow(ia.array_nor(bgImg), cmap='gray', vmin=0, vmax=0.5, interpolation='nearest')

f2 = plt.figure(figsize=(10, 10))
ax2 = f2.add_subplot(111)
ax2.imshow(np.zeros(bgImg.shape, dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')

i = 0
for retain_cell in retain_cells:
    cells[retain_cell].plot_binary_mask_border(plotAxis=ax, color=colors[i], borderWidth=1)
    cells[retain_cell].plot_binary_mask_border(plotAxis=ax2, color=colors[i], borderWidth=1)
    i += 1
plt.show()

# save figures
pt.save_figure_without_borders(f, os.path.join(save_folder, prefix+'_2P_refined_ROIs_with_background.png'), dpi=300)
pt.save_figure_without_borders(f2, os.path.join(save_folder, prefix+'_2P_refined_ROIs_without_background.png'), dpi=300)

# save h5 file
save_file = h5py.File(save_file_name, 'w')
i = 0
for retain_cell in retain_cells:
    print retain_cell, ':', cells[retain_cell].get_binary_area()

    currGroup = save_file.create_group('cell' + ft.int2str(i, 4))
    currGroup.attrs['name'] = retain_cell
    roiGroup = currGroup.create_group('roi')
    cells[retain_cell].to_h5_group(roiGroup)
    i += 1

for attr, value in dfile.attrs.iteritems():
    save_file.attrs[attr] = value

save_file.close()
dfile.close()



