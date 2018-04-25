import numpy as np
import cv2
import tifffile as tf
import os
import scipy.ndimage as ni


def split_bin(arr,downsample=0.5,split=True):
    '''Spatially downsample an image (loaded as array)
    
    Supports spatially bin a very large array/Tiff by first splitting it into 2 and then spatially binning by 'downsample amount'
    Useful for bigTIFF files of large resolution that can't readily be opened in other viewing programs. Use CV2 'nearest' interpolation. 
    
    :param arr: very large image array you wish to spatial downsample (bin)
    :param downsample: amount of downsampling (default =0.5, spatial 2x2 binning)
    :param split: whether to first split the array into two
    
    '''

    if len (arr.shape) !=2:
        raise ValueError, 'Spatial Binning only supports 2D images. Please reformat or run in loop.'
        
    #to get around size restrictions, split array in two first if needed
    if split==True:
        arr_a,arr_b=np.array_split(arr,2)
        #down_a=ia.zoom_image(arr_a,downsample,interpolation ='cubic')
        
        down_a=cv2.resize(arr_a.astype(np.float),dsize=(int(arr_a.shape[1]*downsample),int(arr_a.shape[0]*downsample)),interpolation=cv2.INTER_NEAREST)
        
        down_b=cv2.resize(arr_b.astype(np.float),dsize=(int(arr_b.shape[1]*downsample),int(arr_b.shape[0]*downsample)),interpolation=cv2.INTER_NEAREST)
        
        return np.vstack((down_a,down_b))
    
    elif split==False:
        down=cv2.resize(arr.astype(np.float),dsize=(int(arr.shape[1]*downsample),int(arr.shape[0]*downsample)),interpolation=cv2.INTER_NEAREST)
        
        return down
        

def get_traces_from_masks (roi_mask, movie_arr):
	'''Get a trace from a movie array using a binary mask to select a region of interest
	#param roi_mask: a binary mask as a numpy array
	#param movie file: numpy array of a 3D movie file (X,y,t)

	returns trace
    '''

	roi=ROI(roi_mask)
	trace=roi.get_binary_trace_pixelwise(movie_arr)

	return trace


def imagej_ROI_traces(input_folder, movie_file, roi_file_prefix='Roi', return_masks=True, return_mean_img=False):
	'''Import a folder of imageJ-saved ROI masks (tif format) and extract the masks and corresponding traces.
    
    #param input_folder: folder containing imageJ ROI MASKS
    #param moviefile: movie file that the traces should be extracted from (can be a location specifiying a movie or a movie loaded by tiffile, in numpy format)
    #param roi_file_prefix: the filename prefix that specifies the tif files in the input_folder are ROI files
    #param return_masks: 
    '''

	#check to see if the moviefile is a path rather than an array. Note that if a path is specified,this loads the movie into memory (watch out if movie is large)
	if isinstance(movie_file,str)==True:
		movie_file=tf.imread(movie_file)


	os.chdir(input_folder)

	roi_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f[0:len(roi_file_prefix)] == roi_file_prefix]
	if len(roi_paths)==0:
		print ('no ROI Tif masks found, be sure that each ROI mask starts with the given prefix')
	roi_paths.sort()

	#load each mask as numpy array into a list
	masks=[]
	for m in roi_paths:
		masks.append(tf.imread(m))

	#extract traces from each ROI
	traces=[]

	for t in range (len(roi_paths)):
		trace=get_traces_from_masks(masks[t],movie_file)
		traces.append(trace)
	
	masks=np.asarray(masks)

	#return the corresponding traces and/or masks, t-projection
	if return_masks==True and return_mean_img==True:
		avg_tproj=np.mean(movie_file, axis=0)
		return traces,masks,avg_tproj

	elif return_masks==True and return_mean_img==False:
		return traces,masks

	elif return_masks==False and return_mean_img==True:
		avg_tproj=np.mean(movie_file, axis=0)
		return traces, avg_tproj

	else:
		return traces


class ROI (object):
    
    def __init__(self, roi):
        if len(roi.shape)!=2: raise ValueError, 'ROI is not 2-D. Please reformat'
        
        self.dimension = roi.shape
        self.pixels = np.where(np.logical_and(roi!=0, ~np.isnan(roi)))
        
    def __str__(self):
        return 'ROI object'
        
    def get_binary_trace_pixelwise(self, movie_arr):
        #From Jun 
        #return trace of this ROI (binary format, 0s and 1s) in a given movie
        
        pixels = np.array(self.pixels).transpose()
        trace = np.zeros(movie_arr.shape[0], dtype=np.float32)
        for pixel in pixels:
            trace = trace + movie_arr[:, int(pixel[0]), int(pixel[1])].flatten().astype(np.float32)
    
        return trace/len(self.pixels[0])
    
    
def generate_surround_masks(rois, width=15, exclusion=5):
	'''generates surround/neuropil masks by dilation of an ROI

	#param rois: array of binary mask ROIs to calculate surround/neuropil masks from
	#parm width: number of dilation steps to perform (larger=more space in the surround)
	#param exclusion: number of dilation steps to exclude from near the ROI. Useful for have a spacer between the ROI and the surround

	#returns array of neuropil_masks for each ROI
	'''

	roi_masks=np.array(rois)

	neuropil_masks=np.empty(roi_masks.shape, np.bool)

	for masks in range(neuropil_masks.shape[0]):
		#generate the neuropil masks
		neuropil_masks[masks] = ni.binary_dilation(roi_masks[masks], iterations=width) - \
		ni.binary_dilation(roi_masks[masks], iterations=exclusion)

		neuropil_masks[masks] = neuropil_masks[masks] - np.logical_and(neuropil_masks[masks], np.sum(roi_masks, axis=0))


	return neuropil_masks



