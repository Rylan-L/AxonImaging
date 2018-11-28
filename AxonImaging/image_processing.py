import numpy as np
import cv2
import tifffile as tf
import os
import scipy.ndimage as ni
import matplotlib.pyplot as plt
from skimage.measure import label as sk_label
from skimage.measure import regionprops


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


class weighted_ROI(ROI):

    def __init__(self, mask):
        super(weighted_ROI,self).__init__(mask)
        self.weights = mask[self.pixels]

    def get_weighted_trace_pixelwise(self, mov, is_area_weighted=False):
        '''
        return trace of this ROI in a given movie, the contribution of each pixel in the roi was defined by its weight
        :param is_area_weighted: bool, if False, total area of the mask is calculated in a binary fashion
                                       if True, total area of mask is calculated in a weighted fashion
        calculation is done in pixelwise fashion
        '''
        pixels = self.get_pixel_array()
        trace = np.zeros(mov.shape[0], dtype=np.float32)
        for i, pixel in enumerate(pixels):
            # trace += mov[:, pixel[0], pixel[1]]  # somehow this is less precise !! do not use
            trace = trace + self.weights[i] * (mov[:, pixel[0], pixel[1]]).astype(np.float32)
        # print trace
        if not is_area_weighted:
            return trace / self.get_binary_area()
        elif is_area_weighted:
            return trace / self.get_weight_sum()
        else:
            raise ValueError('is_area_weighted should be a boolean variable.')

    def get_binary_mask(self):
        '''
        generate binary mask of the ROI, return 2d array, with 0s and 1s, dtype np.uint8
        '''
        mask = np.zeros(self.dimension,dtype=np.uint8)
        mask[self.pixels] = 1
        return mask
    
    def to_h5_group(self, h5Group):
        '''
        add attributes and dataset to a h5 data group
        '''
        h5Group.attrs['dimension'] = self.dimension
        h5Group.attrs['description'] = str(self)
        if self.pixelSizeX is None: h5Group.attrs['pixelSize'] = 'None'
        else: h5Group.attrs['pixelSize'] = [self.pixelSizeY, self.pixelSizeX]
        if self.pixelSizeUnit is None: h5Group.attrs['pixelSizeUnit'] = 'None'
        else: h5Group.attrs['pixelSizeUnit'] = self.pixelSizeUnit

        dataDict = dict(self.__dict__)
        _ = dataDict.pop('dimension');_ = dataDict.pop('pixelSizeX');_ = dataDict.pop('pixelSizeY');_ = dataDict.pop('pixelSizeUnit')
        for key, value in dataDict.iteritems():
            if value is None: h5Group.create_dataset(key,data='None')
            else: h5Group.create_dataset(key,data=value)

    
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

def downsample_time(movie_path, downsample_by=10, resave_as=''):
    '''Downsamples a movie in time by averaging (mean) in chunks. Takes input as h5, tif, or numpy array

    #param movie_path: movie as array, h5 file, or tiff file
    #param downsample_by: how many frames to average together
    #param resave_as: whether to resave the dowsampled movie as h5 or tiff .Leave as '' to not save anything.

    #returns downsampled array (if not resaving)
    '''

    #if the file_path is an h5, downsample in chunks by reading each from disk (saves on memory)
    if movie_path[-2:] == '.h5':
        h5_file= h5py.File(movie_path,'r')
        #if h5, currently assumes the movie is stored as the key 'data'
        movie=h5_file['data']
        new_file_name=os.path.basename(movie_path)+'_downsampled_by_'+str(downsample_by)

    elif movie_path[-3]=='.tif' or movie_path[-4]=='.tiff':
        movie=tf.imread(movie_path)
        new_file_name=os.path.basename(movie_path)+'_downsampled_by_'+str(downsample_by)

    elif type(movie_path) is np.ndarray:
        movie=movie_path
        new_file_name='movie'
    else:
        raise ('Movie path does not refer to a valid movie type (h5,tif,or array)')


    total_frames=movie.shape[0]
    print ('\n \n total number of movie frames is ' + str(total_frames))
    final_num_frames = total_frames //downsample_by
    print ('\n new movie will be '+str(final_num_frames) +' frames')
        
    down_list=[]
    for ii in range(final_num_frames):
        chunk=movie[(ii*downsample_by):((ii+1)*downsample_by),:,:].astype(np.float)
        avg_frame=np.mean(chunk,axis=0)
        down_list.append(avg_frame)

    #resave downsampled movie as tif or H5

    if resave_as=='tif':
        print ('Saving downsampled movie as tif.....')
        tf.imsave(new_file_name+'.tif',np.array(down_list))

    elif resave_as=='h5':
        #or resave downsampled movie H5

        down_h5 = h5py.File(new_file_name+'.h5', 'w')
        down_h5.create_dataset('data', data=np.array(down_list))
        down_h5.close()

    elif resave_as=='':
        return np.array(down_list)


def concenate_tifs_folder(path,save=True):
    '''
    #param path: folder path that contains a list of labeled tifs that can be sorted numerically by their names (ie 1-10)
    #param save: whether to save the concentated tifs to disk as a single large tif 
    '''
   
    os.chdir(path)

    file_list = [f for f in os.listdir(path) if f[-3:] == 'tif']
    file_list.sort()
    
    print '\n'.join(file_list)
    
    mov = []
    
    for n in file_list:
     
        mov.append(tf.imread(n))
        
    mov = np.concatenate(mov, axis=0)
    
    if save==True:
        f_name, ext = os.path.splitext(n)
        tf.imsave(f_name + '_concatenated' + ext, mov)
    
    else:
        return mov


def int2str(num,length=None):
    '''
    generate a string representation for a integer with a given length
    :param num: non-negative int, input number
    :param length: positive int, length of the string
    :return: string represetation of the integer
    '''

    rawstr = str(int(num))
    if length is None or length == len(rawstr):return rawstr
    elif length < len(rawstr): raise ValueError, 'Length of the number is longer then defined display length!'
    elif length > len(rawstr): return '0'*(length-len(rawstr)) + rawstr
    

def get_masks(labeled, minArea=None, maxArea=None, isSort=True, keyPrefix = None, labelLength=None):
    '''
    From Jun Zhuang's Cortical Mapping Package


    get mask dictionary from labeled map (labeled by scipy.ndimage.label function), masks with area smaller than
    minArea and maxArea will be discarded.

    :param labeled: 2d array with non-negative int, labelled map (ideally the output of scipy.ndimage.label function)
    :param minArea: positive int, minimum area criterion of retained masks
    :param maxArea: positive int, maximum area criterion of retained masks
    :param isSort: bool, sort the masks by area or not
    :param keyPrefix: str, the key prefix for returned dictionary
    :param labelLength: positive int, the length of key index

    :return masks: dictionary of 2d binary masks
    '''

    maskNum = np.max(labeled.flatten())
    masks = {}
    for i in range(1, maskNum + 1):
        currMask = np.zeros(labeled.shape, dtype=np.uint8)
        currMask[labeled == i] = 1

        if minArea is not None and np.sum(currMask.flatten()) < minArea:
            continue
        elif maxArea is not None and np.sum(currMask.flatten()) > maxArea:
            continue
        else:
            if labelLength is not None:
                #mask_index = ft.int2str(i, labelLength)
               
               
                mask_index= '0'*(labelLength-len(str(i))) + str(i)
                
            else:
                mask_index = str(i)

            if keyPrefix is not None:
                currKey = keyPrefix + '_' + mask_index
            else:
                currKey = mask_index
            masks.update({currKey: currMask})

    if isSort:
        
        masks = sort_masks(masks, keyPrefix=keyPrefix, labelLength=labelLength)

    return masks

def sort_masks(masks, keyPrefix=None, labelLength=3):
    '''
    From Jun Zhuang's Cortical Mapping Package
    
    sort a dictionary of binary masks, big to small
    '''

    maskNum = len(masks.keys())
    order = []
    for key, mask in masks.iteritems():
        order.append([key,np.sum(mask.flatten())])

    order = sorted(order, key=lambda a:a[1], reverse=True)

    newMasks = {}
    for i in range(len(order)):
        if keyPrefix is not None: currKey = keyPrefix+'_'+'0'*(labelLength-len(str(i))) + str(i)
        else: currKey = int2str(i,labelLength)
        newMasks.update({currKey:masks[order[i][0]]})
    return newMasks

def plot_mask_borders(mask, plot_axis=None, color='#ff0000', border_width=2, **kwargs):
    """
    Modified from Jun Zhuang, simplified version

    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """
    if not plot_axis:
        f = plt.figure()
        plot_axis = f.add_subplot(111)

    plot_mask = np.ones(mask.shape, dtype=np.uint8)

    plot_mask[np.logical_or(np.isnan(mask), mask == 0)] = 0

    currfig = plot_axis.contour(plot_mask, levels=[0.5], colors=color, linewidths=border_width, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plot_axis.get_ylim())
    y_lim.sort()
    plot_axis.set_ylim(y_lim[::-1])

    plot_axis.set_aspect('equal')

    return currfig


def full_file_path(directory, file_type='.tif',prefix='', exclusion='Null'):
    """Returns a list of files (the full path) in a directory that contain the characters in "prefix" with the given extension, while excluding files containing the "exclusion" characters"""

    file_ext_length=len(file_type)
    for dirpath,_,filenames in os.walk(directory):
      return [os.path.abspath(os.path.join(dirpath, filename)) for filename in filenames if prefix in filename and exclusion not in filename and filename[-file_ext_length:] == file_type ]

def avi_from_mov(mov,output_name='movie.avi', codec='XVID', normalize_histogram=False, FR=30.) :

    mov=mov.astype(np.float32)    
    
    writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"ffds"), FR, (512, 512), False)
    
    mov=(mov - np.amin(mov)) / (((np.amax(mov) - np.amin(mov))))
    
    
    mov=(mov*255).astype(np.uint8)
    
    for xx in range(np.shape(mov)[0]):

        if normalize_histogram:
            #writer.write(cv2.equalizeHist(mov[xx,:,:]))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(mov[xx,:,:])

            writer.write(cl1)
        else:
            writer.write(mov[xx,:,:])     
     
    writer.release()
 
    print ('AVI movie Successfully Created')

def measure_eccentricity(roi):
    #measure the eccentricity of a binary ROI
   
    roi[roi > 0] = 1
    labeled_roi=sk_label(roi)
    regions = regionprops(labeled_roi)

    return regions[0]['eccentricity']