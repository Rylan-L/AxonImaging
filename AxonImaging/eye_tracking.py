# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 12:44:08 2018

@author: rylanl

Uses code from internal AllenSDK authored by Michael Buice, where noted.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math



def calculate_pupil_diameter(pupil_params, scale_factor):
    '''
    Param: pupil parameters: array of pupil parameters [x, y, angle, axis1, axis2]
    Param: scale_factor: scale factor for number of pixels per millimeter (mm). For example, 1 pixel = 1 micron then = 0.001
    
    Assumes the pupil is circular
    '''
       
    #calculate area of ellipse from A= π * axis a * axis b. multiply by the scaling factor to get into mm
    elipse_area = np.pi * np.multiply(pupil_params[:,3] * scale_factor, pupil_params[:,4] * scale_factor)
    
    #solve for the diameter of a circle with the same area as the calculated elipse above
    diameter_eye = elipse_area**0.5 / np.pi * 2

    return diameter_eye

def replace_nan_with_last_good_value(signal):  
    '''
    Replaces all NaN values in a 1-D array with the previous non-NaN (good value)
    :Param signal: 1-D array of values, some of which contain NaN
    
    :return signal array, lacking NaN values
    '''
    signal_clean = signal
    
    #for initialization, set the last good value to the first value
    last_good_value=signal[0]

    for sample in range(np.shape(signal)[0]):
        bad=False
        if math.isnan(signal[sample]):
            bad=True
           
        if bad:
            signal_clean[sample]= last_good_value
           
        else:
            signal_clean[sample] = signal[sample]
            last_good_value = signal[sample]
        
    return signal_clean

def remove_pupil_outliers (pupil_params, pupil_percentile):
    #calculate the pupil area and set values greater than the threshold using NaN (code chunk from Allen Brain Observatory)
    area = np.pi*pupil_params.T[3]*pupil_params.T[4]
    threshold = np.percentile(area[np.isfinite(area)], pupil_percentile)
    outlier_index = area > threshold
    pupil_params[outlier_index, :] = np.nan
    return pupil_params
    

def post_process_pupil(pupil_params, pupil_percentile, replace_nan=False, smooth_filter_sigma=0):
    '''Filter pupil parameters and replace outliers with nan.

    :Param pupil_params: array of pupil parameters [x, y, angle, axis1, axis2]
    :Param pupil_percentile: percentile for thresholding outliers. Pupil area values higher than this value are set to NaN.
    :Param replace_nan: Optional, Boolean, whether to replace NaN values (outliers) with the last non-NaN, good value. 
    :Param smooth_filter_sigma: Optional, whether to guassian filter the data and which sigma to use. Recommended is a multiple of the sample rate of the signal (eg 0.05*sample_freq).
                                If zero, the trace is not filtered.
    
    Return: pupil parameters with outliers replaced with nan and optionally guassian filtered
    '''
    
    #first calculate the pupil area and set values greater than the threshold using NaN 
    
    pupil_params=remove_pupil_outliers(pupil_params, pupil_percentile)
    
    #optionally interpolate across NaN values
    if replace_nan:
        pupil_no_nan = np.zeros(np.shape(pupil_params))
        
        for xx in range (np.shape(pupil_params)[1]):
            pupil_no_nan[:,xx]=replace_nan_with_last_good_value(pupil_params[:,xx])
        
        pupil_params=pupil_no_nan
    
    #optionally guassian filter
    if smooth_filter_sigma!=0:
        
        pupil_filtered = np.zeros(np.shape(pupil_params))
        
        for yy in range (np.shape(pupil_params)[1]):
            pupil_filtered[:,yy]=gaussian_filter1d(pupil_params[:,yy], int(smooth_filter_sigma))
       
        pupil_params=pupil_filtered
        
    return pupil_params


def post_process_cr(cr_params, replace_nan=False,smooth_filter_sigma=0.0):
    """Main body of code From Allen Brain Obsevatory SDK
    
    This will replace questionable values of the CR x and y position with 'nan'

        1)  threshold ellipse area by 99th percentile area distribution
        2)  median filter using custom median filter
        3)  remove deviations from discontinuous jumps

        The 'nan' values likely represent obscured CRs, secondary reflections, merges
        with the secondary reflection, or visual distortions due to the whisker or
        deformations of the eye"""

    area = np.pi*cr_params.T[3]*cr_params.T[4]

    # compute a threshold on the area of the cr ellipse
    dev = median_absolute_deviation(area)
    if dev == 0:
        print("Median absolute deviation is 0,"
                        "falling back to standard deviation.")
        dev = np.nanstd(area)
    threshold = np.nanmedian(area) + 3*dev

    x_center = cr_params.T[0]
    y_center = cr_params.T[1]

    # set x,y where area is over threshold to nan
    x_center[area>threshold] = np.nan
    y_center[area>threshold] = np.nan

    # median filter
    x_center_med = medfilt_custom(x_center, kernel_size=3)
    y_center_med = medfilt_custom(y_center, kernel_size=3)

    x_mask_finite = np.where(np.isfinite(x_center_med))[0]
    y_mask_finite = np.where(np.isfinite(y_center_med))[0]

    # if y increases discontinuously or x decreases discontinuously,
    #  that is probably a CR secondary reflection
    mean_x = np.mean(x_center_med[x_mask_finite])
    mean_y = np.mean(y_center_med[y_mask_finite])

    std_x = np.std(x_center_med[x_mask_finite])
    std_y = np.std(y_center_med[y_mask_finite])

    # set these extreme values to nan
    #x_center_med[x_center_med < mean_x - 3*std_x] = np.nan
    #y_center_med[y_center_med > mean_y + 3*std_y] = np.nan
    x_center_med[np.abs(x_center_med - mean_x) > 3*std_x] = np.nan
    y_center_med[np.abs(y_center_med - mean_y) > 3*std_y] = np.nan

    either_nan_mask = np.logical_and(np.isnan(x_center_med),np.isnan(y_center_med))
    x_center_med[either_nan_mask] = np.nan
    y_center_med[either_nan_mask] = np.nan

    new_cr = np.vstack([x_center_med, y_center_med]).T

    #optionally interpolate
    if replace_nan:
        cr_no_nan = np.zeros(np.shape(new_cr))
        
        for xx in range (np.shape(new_cr)[1]):
            cr_no_nan[:,xx]=replace_nan_with_last_good_value(new_cr[:,xx])
        
        new_cr=cr_no_nan
        
    #optionally guassian filter
    if smooth_filter_sigma!=0.0:
        from scipy.ndimage import gaussian_filter1d
        
        cr_filtered = np.zeros(np.shape(new_cr))
        
        for yy in range (np.shape(new_cr)[1]):
            cr_filtered[:,yy]=gaussian_filter1d(new_cr[:,yy], int(smooth_filter_sigma))
       
        new_cr=cr_filtered

    return new_cr



def eye_pos_degrees(pupil,cr,eye_radius=1.7,pixel_scale=.001, relative_to_mean=False):
    '''
    Calculates azimuth and altitude positions for the eye in terms of units of degrees.
    
    :param pupil: x and y axis measurements for the centroid of the pupil
    :param cr: x and y axis measurements for the centroid of the corneal reflection
    :param eye_radius: assumed radius of the eye, in mm. Based on Remtulla & Hallett, 1985 
    :param pixel_scale: scaling value for number of mm per pixel from the eye monitoring movie in terms of mm per pixel. Requires calibration based on camera magnification, resolution, etc
    :param relative_to_mean: whether to return values that are relative (subtracted from) to the mean pupil/CR for the X and Y axes. If False, returns values not relative to the mean.
    
    :return: azimuth and altiude eye position in measurements of degrees
    
    Based on methods described in Zhuang et al, 2017 and Denman et al, 2017
    
    for azimuth negative should equal nasal (worth checking)
    '''

    #calculate x and y positions as the difference between the pupil and corneal reflection centroids
    #note that the measurments need to have the same units, so multiply by pixel scale to get in terms of mm (same as eye radius)
    
    x=(pupil[:,0]-cr[:,0])*pixel_scale
    
    y=(pupil[:,1]-cr[:,1])*pixel_scale

    #if relative to the mean, calculate the mean position and subtract measurements from it
    if relative_to_mean==True:
        #np.arctan returns in units of [-pi/2, pi/2], multiply by 180 over pi to get into degrees (alternatively use math.degrees)
        delta_x=np.subtract(x,np.nanmean(x))
        delta_y=np.subtract(y,np.nanmean(y))

    
        azi=np.arctan(np.divide(delta_x,eye_radius))*(180./np.pi)
        alt=np.arctan(np.divide(delta_y,eye_radius))*(180./np.pi)

    else:
        #np.arctan returns in units of [-pi/2, pi/2], multiply by 180 over pi to get to degrees
        azi=np.arctan(np.divide(x,eye_radius))*(180./np.pi)
        alt=np.arctan(np.divide(y,eye_radius))*(180./np.pi)
    
   
    return azi,alt


def median_absolute_deviation(a, consistency_constant=1.4826):
    '''From Allen Brain Observatory SDK
    
    Calculate the median absolute deviation of a univariate dataset.

    Parameters
    ----------
    a : numpy.ndarray
        Sample data.
    consistency_constant : float
        Constant to make the MAD a consistent estimator of the population
        standard deviation (1.4826 for a normal distribution).

    Returns
    -------
    float
        Median absolute deviation of the data.
    '''
    return consistency_constant * np.nanmedian(np.abs(a - np.nanmedian(a)))

def medfilt_custom(x, kernel_size=3):
    '''From Allen Brain Observatory SDK
    
    This median filter returns 'nan' whenever any value in the kernal width is 'nan' and the median otherwise'''
    T = x.shape[0]
    delta = kernel_size/2

    x_med = np.zeros(x.shape)
    window = x[0:delta+1]
    if np.any(np.isnan(window)):
        x_med[0] = np.nan
    else:
        x_med[0] = np.median(window)

    # print window
    for t in range(1,T):
        window = x[t-delta:t+delta+1]
        # print window
        if np.any(np.isnan(window)):
            x_med[t] = np.nan
        else:
            x_med[t] = np.median(window)

    return x_med

def eye_pos_gaze_degrees(pupil,cr,eye_radius=1.7,pixel_scale=.001):
    ''' Based on Denman et al, 2017
    Returns a single degree measurement for change in eye position in degrees relative to the mean
    
    :param pupil: x and y axis measurements of pupil 
    :param cr: x and y axis measurements of corneal reflection
    :param eye_radius: assumed radius of the eye in mm
    :param pixel_scale: scaling value for number of mm per pixel from the eye monitoring movie.
    
    :return: trace array for eye movements
    '''
    
    x=pupil[:,0]-cr[:,0]
    
    y=pupil[:,1]-cr[:,1]
    
    delta_x=np.subtract(x,np.nanmean(x))
        
    delta_y=np.subtract(y,np.nanmean(y))
    
    summed_square=np.add(np.square(delta_x),np.square(delta_y))*pixel_scale
        
    delta_i=np.sqrt(summed_square)
        
    radian_delta_i=np.arctan(np.divide(delta_i,eye_radius))
    #convert to degrees
    degree_delta_i=radian_delta_i*180/np.pi
    
    return degree_delta_i
