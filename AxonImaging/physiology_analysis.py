#Author: Rylan Larsen
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.signal import convolve, boxcar

from signal_processing import threshold_greater

    
    
def get_processed_running_speed (vsig,vref_mean,sample_freq, smooth_filter_sigma = 0.05, wheel_diameter = 16.51, positive_speed_threshold= 50, negative_speed_threshold= -5):
    ''' Returns the running speed given voltage changes from an encoder wheel. Speeds are smoothed and outlier
    values above or below arbrituarly defined thresholds are set as NaN. 
    
    :param Vsig: voltage signal which changes as a function of wheel movement (running)
    :param Vref_mean: reference voltage (typically 5V +/- small offset that is encoder dependent. Best way to calculate has been to use np.median(vref[np.abs(vref)<20]) 
    :param sample_freq: sampling frequency which Vsig and Vref are acquired at
    :param smooth_filter_sigma: value used for guassian filtering 
    :param wheel_diameter: diameter of running wheel
    :param positive_speed_threshold: maximum allowed positive speed (sets impossibly high running speeds equal to NaN)
    :param negative_speed_threshold: maximum allowed negative speed (sets impossibly high backwards running speeds equal to NaN)
    :param  units: whether to return in terms of seconds (dependent on the passed-in sample freq) or samples
    :return: smooth traced of running speed in cm/s per sample with outliers set to NaN
    '''
        
    position_arc = vsig*(2.*np.pi)/vref_mean 
    position_arc_smooth = gaussian_filter1d(position_arc, int(smooth_filter_sigma*sample_freq))
    speed_arc = np.append(np.diff(position_arc_smooth),0) * sample_freq
    speed = speed_arc * wheel_diameter
    speed_smooth = np.copy(speed)
    speed_smooth[np.logical_or(speed>=positive_speed_threshold,speed<=negative_speed_threshold)]=np.nan
    
    mask = np.isnan(speed_smooth)
    mask2 = np.zeros(mask.shape, dtype=np.bool)

    for n,p in enumerate(mask):
        if p:
            mask2[(n-(2*int(smooth_filter_sigma*sample_freq))):(n+int((2*smooth_filter_sigma*sample_freq+1)))] = True # extend mask 2 filter widths to extend interpolation

                    
    speed_smooth[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), speed[~mask2])


    return speed_smooth
    

def get_auditory_onset_times(microphone, sample_freq, threshold=1, stdev_samples=10,filter_width=20):
    '''
    Finds the onset of an auditory event through first calculating a standard deviation across user defined samples and then thresholding the stdeviations to find the onset times.
    
    :param microphone: an analog microphone signal
    :param samplefreq: the sampling frequency at which the auditory signal was acquired at
    :param threshold = threshold value in units of standard deviation for finding onset times (values above this are marked as a valid onset)
    :param stdev_samples=number of samples to calculate each standard deviation from.
    :
    :return: the onset sound_times in the units of seconds
    '''

    #get the standard deviation across user-defined number of samples
    step=int(stdev_samples)
    stdev=[]
    for ii in range(0,microphone.shape[0],step):
        chunk=microphone[ii:ii+step]
        stdev.append(np.std(chunk))

    
    if filter_width>0:
        stdev_filtered=convolve(stdev, boxcar(M=filter_width))

    else:
        print('using un-filtered stdev of signal')
        stdev_filtered=stdev

    #get the up samples #s through thresholding
    stamps=threshold_greater(np.array(stdev_filtered),threshold)

    #multiply these samples # by user-defined number of stdev_samples to account for the downsampling that occured when the standard deviation was calculated
    stamps=np.multiply(stamps,stdev_samples)

    sound_times = np.divide(stamps,sample_freq)
    print ('total number of sound presentations found = '+ str(len(sound_times)))
    return sound_times





def microphone_to_dB (signal, sensitivity=250, pre_amp_gain=12):
    ''' Converts microphone voltage to decibels given the microphone sensitivity and pre amp gain.
    :param signal: the analog microphone voltage (in V)
    :param sensitivity: the sensitivity of the microphone in mv/Pa
    :param pre_amp_gain: gain setting on the microphone pre amp (in dB)
    '''
    
        #reference is "threshold for hearing" 20 micropascals at 1 kHz, also called SPL
    reference=20E-6
    baseline_v=reference*sensitivity
    
    db=np.log10((signal/baseline_v))*20
    
    db_nogain=db-pre_amp_gain
    
    return db_nogain
        
    
    #convert signal to pascals
    
    #divide by the preamp gain, multiply by 1000 to convert from volts to mV
 
    
    #divide by the microphone sensitivity in mV/Pa
  
      
    #dB equation from voltage is 20 * log ()

def shift_aud_frames_by_mic_delay(mic_onsets, aud_frames, vsync):
    '''
    Time aligns auditory stimulation onset times that are given in terms relative to monitor frames (ie auditory stimulation was 
    presented on frame 50-100) into accurate times for which they are played/heard (detected on a microphone).
    
    Requires that the number of sound times presented is the same quantity as the number detected by the microphone
    
    :param mic_onsets: auditory onsets detected by a microphone (see get_auditory_onset_times function) (seconds)
    :param aud_frames: frames when the auditory stimulation was initiated (typically from pikl file) (frame #'s)
    :param vsync: times of each monitor frame presentation on the same time base as the microphone (seconds)
    
    :return: array of frame numbers that correspond to onset of the auditory stimulation being played 
          
    '''
    
    #compare total number of auditory stims with the expected number of presentations.
    if len(mic_onsets)==len(aud_frames):
        #get the auditory stimulation time from the pickle file and convert it to a Vsync time
           
        
        #get the auditory stimulation time from the pickle file and convert it to a Vsync time

        sound_frames=[]
        for ii in range(len(aud_frames)):
            #calculate the difference in time between the detection (microphone) and the presentation, in terms of seconds
            dif=mic_onsets[ii]-vsync[aud_frames[ii]].astype(np.float32)
    
            presented_time=vsync[aud_frames[ii]]+dif
            #find the vysnc time that most closely matches the stimulation
            index=np.argmin(np.abs(vsync-presented_time))
            sound_frames.append(index)
            #print ('time of presentation '+ str(vsync[aud_onsets[ii]]) + ' time of detection ' + str(sound_times[ii]))

        sound_frames=np.array(sound_frames)
        print ('mean number of visual frames between presentation and detection is ' + str((np.mean(sound_frames-aud_frames)))) + ' frames or '+ str((1/np.median(np.diff(vsync))*(np.mean(sound_frames-aud_frames))))+' millseconds (assuming 60 fps)'
        
        return sound_frames              
    else:
        print ('Number of known auditory presentations '+str(len(aud_frames))+ ' does not equal those detected by microphone '+ str(len(mic_onsets)))
        return
    



