#Author: Rylan Larsen

import numpy as np
from axonimaging.signal_processing import stim_df_f, significant_response,butter_lowpass_filter

import os
import h5py
import pandas as pd
    


def threshold_greater(data, threshold=0):
    #from Jun Zhuang
    pos = data > threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0]


def last_pos_above_thresh(signal, threshold):
    '''finds array position greater than a threshold where the next position is less than the threshold. Useful for finding
    'useful for finding transitions from above a threshold to their end
    
    :param pre_thresh_samp Number of before a threshold crossing that must be below the threshold to consider it a threshold crossing, default=1
    '''
    return np.flatnonzero((signal[:-1] > threshold) & (signal[1:] < threshold))

def last_pos_above_low_thresh(signal, threshold, low_threshold ):
    '''finds array position greater than a threshold where the next position is less than a different, low threshold. 
    Useful for finding for finding high transitions to an arbituary low value
    
    :param signal: The trace that you wish to extract threshold transitions from
    :param threshold: the 'high' threshold. The threshold value that the signal must be above inititially. 
    :param low_threshold: the 'low' threshold
    :param pre_thresh_samp: Number of before a threshold crossing that must be below the threshold to consider it a threshold crossing, default=1
    :param post_thresh_samp: Number of samples after a threshold crossing that must be below the threshold to consider it below the crossing, default=1
    
    '''
    return np.flatnonzero((np.mean(signal[:-1]) > threshold) & (np.mean(signal[1:]) < low_threshold))


def last_pos_below_thresh(signal, threshold,pre_thresh_samp=1):
    '''finds array position less than a threshold where the next position is above than the threshold. Useful for finding
    useful for finding transitions from below a threshold to above
    
    :param signal: The trace that you wish to extract threshold transitions from
    :param pre_thresh_samp Number of before a threshold crossing that must be below the threshold to consider it a threshold crossing, default=1
    '''
    
    return np.flatnonzero((signal[:-1] < threshold) & (signal[1:] > threshold))



def get_values_from_event_times (signal_trace, signal_ts, events,total_dur=0,relative_time=0):
    '''Get values from a digitally-sampled trace given event times. Useful for extracting a fluorescence value from a 
    2-P trace given event times ('what is the fluorescence value at event time x)
    
    :param signal_trace: The trace that you wish to extract a value from at a given event time
    :param signal_ts: the digital time stamps corresponding to the signal_trace
    :param events: a list of event times
    :param total_dur: the total duration to sample from after the onset (+/- relative time). If zero, returns a single timepoint
    :param relative_time: time relative to each event onset for extracting values (seconds, can be negative)
    
    '''
    values=[]
    
    for xx in range(len(events)):
        index=np.argmin(np.abs((signal_ts+relative_time)-events[xx]))
        
        if total_dur>0:
            frame_dur=1/np.mean(np.diff(signal_ts))
            values.append(signal_trace[index:(index+int(round(total_dur*frame_dur)))])
        
        else:
            values.append(signal_trace[index])
    return values
        
 

def get_processed_running_speed (vsig,vref,sample_freq, smooth_filter_sigma = 0.05, wheel_diameter = 16.51, positive_speed_threshold= 70, negative_speed_threshold= -5):
    ''' Returns the running speed given voltage changes from an encoder wheel. Speeds are smoothed and outlier
    values above or below arbrituarly defined thresholds are set as NaN. 
    
    :param Vsig: voltage signal which changes as a function of wheel movement (running)
    :param Vref: reference voltage (typically 5V +/- small offset that is encoder dependent
    :param sample_freq: sampling frequency which Vsig and Vref are acquired at
    :param smooth_filter_sigma: value used for guassian filtering 
    :param wheel_diameter: diameter of running wheel
    :param positive_speed_threshold: maximum allowed positive speed (sets impossibly high running speeds equal to NaN)
    :param negative_speed_threshold: maximum allowed negative speed (sets impossibly high backwards running speeds equal to NaN)
    :param  units: whether to return in terms of seconds (dependent on the passed-in sample freq) or samples
    :return: smooth traced of running speed in cm/s per sample with outliers set to NaN
    '''
    
    from scipy.ndimage import gaussian_filter1d

    vref_mean = np.median(vref[np.abs(vref)<20]) 
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

def threshold_period(signal, threshold, min_low, sample_freq=30.0, irregular_sampled_ts=np.array([]),min_time_between=False, pre_thresh_time=False,low_threshold=False,min_time=0.01, trace_end=994.0, 
                     exclusion_signal=np.array([]), exclusion_thresh=0, exclusion_dur=False, exclusion_sf=30., exclusion_logic='exclude'):
    '''Defines onset times for when a signal crosses a threshold value for an EXTENDED period of time.
    The period of time for the period is defined by min_low, which specifies the minimum amount of time the signal must
    fall below the previously defined threshold to end the period. 
    
    This is useful for calculating event-triggered averages for noisy signals where the noise may make it appear that 
    the period in which the signal has exceeded has the threshold has ended, but actually is just interupted by a noisy or 
    variable period in which the threshold is only temporarily lower than the defined threshold value
    
    :param signal: the signal to process (units of samples)
    :param threshold: the threshold the signal must be greater than to initiate a period (signal units)
    :param min_low: the minimum amount of time that the signal must fall below the threshold value (or low_threshold) to end the period (seconds)
    :param sample_freq: the sampling frequency of the signal (samples/second, aka Hz)
    :param pre_thresh_time: Amount of time before a threshold crossing where signal must be below the threshold to consider it a threshold crossing, default=1 sec
    :param low_threshold:After the trace crosses an upward threshold, low_threshold can be be used to specified a threshold for when the period ends. (signal units)
    :param min_time: minimum amount of a whole period (in time) that is allowed for inclusion (seconds). I.E. only want periods of at least x length.
    :trace_end: the end of the signal region for analysis. Typically this is the end of an imaging session and is included to avoid analyzing data where there is no corresponding images.
    
    :param exclusion_signal: an optional, different array to use as an exclusion (or exclusive inclusive) indicator of 
    
    
    :return list of start, end, and duration times for each period
    '''
    
    end_ind=int(round(trace_end*sample_freq))
       
    #find positions above the threshold (beginning of period). 
    upwards=last_pos_below_thresh(signal[:end_ind],threshold)
    
    #find positions that were above the threshold but the next position is below (end of period). 
    
    if low_threshold==False:
        downwards=last_pos_above_thresh(signal[:end_ind],threshold)
       
    else:
        downwards=last_pos_above_low_thresh(signal[:end_ind],threshold, low_threshold=low_threshold)
    #check for weird situation where the trace starts above the threshold
    if signal[0]>=threshold:
        print ('Warning: the first point in the trace is already above threshold. Discarding the first downcrossing.')
        print(downwards.size)

        try:
            downwards=downwards[1:-1]
            downwards=np.append(downwards, (downwards[-1]+10))
        except:
            print ('Signal appears to start above the threshold and never go below. Therefore, no threshold crossings found')
            return
                
    #check the threshold crossing to make sure an extended period of time prior to the upcrossing that the signal is below
    if pre_thresh_time!=False:
        #check to see if an arbituary number of points before the threshold are below the threshold
        pre_time_samples=int(round(pre_thresh_time*sample_freq))
      
        good_ups=[]
        good_downs=[]
        
        for xx in range(len(upwards)-1):
            mean_pre_cross=np.mean(signal[(upwards[xx]-pre_time_samples):(upwards[xx]-1)])
            if mean_pre_cross<threshold:
                good_ups.append(upwards[xx])
                good_downs.append(downwards[xx])
        
        upwards=np.asarray(good_ups) 
        downwards=np.asarray(good_downs)
   
    #if the signal is not regularly sampled, the absolute time in seconds need to be determined using a separate array of time-stamps
    if irregular_sampled_ts.size==0:
        t_up_conv=upwards/sample_freq
        t_down_conv=downwards/sample_freq
    
    elif irregular_sampled_ts.size!=0:
        t_up_conv=[]
        t_down_conv=[]
        
        #for irregular (time-stamped data) get the actual times from the time-stamps without assuming even sample-spacing.
        #this converts the thresholded signal back into real time by using time_stamps
        for yy in range(len(upwards)-1):
             t_up_conv.append(irregular_sampled_ts[upwards[yy]])
             t_down_conv.append(irregular_sampled_ts[downwards[yy]])
        
        np.asarray(t_up_conv)
        np.asarray(t_down_conv)
   
     
    threshed_epochs_u=[]
    threshed_epochs_d=[]
    
    if exclusion_signal.size==0:
        #define the orginal epochs by zipping them together into tuple
        epoch_o=zip(t_up_conv,t_down_conv)
    else:

        for ii in range(len(t_up_conv)-1):
            
            
            #convert the valid upcrossing time to the units of the exclusion signal
            #note t_up_conv[ii] is in seconds, get exclusion 
            
            if isinstance(exclusion_dur, (tuple)):
                #subtract the first tuple value from the up time to get the beginning. Multiple by the sampling frequency to get in terms of indices of the array
                exclude_start=int(round((t_up_conv[ii]-exclusion_dur[0])*exclusion_sf))
                
                #add the second tuple value from the up time to get the end
                exclude_end=int(round((t_up_conv[ii]+exclusion_dur[1] )*exclusion_sf))     
                
                
                if exclude_start<1:
                    print ('There are not enough time points at the beginning of the exclusion trace to calculate the specified exclusion_dur beginning. Setting the start equal to the first time point')
                    exclude_start=1
                if exclude_end > np.shape(exclusion_signal)[0]:
                    print ('There are not enough time points at the End of the exclusion trace to calculate the specified exclusion_dur ending. Setting the ending equal to the last time point')
                    exclude_end=np.shape(exclusion_signal)[0]
                    

            elif isinstance(exclusion_dur, (int, float)):
                print ('only single duration timepoint passed. Assuming this timepoint refers to a period prior to the onset of the threshold crossing (start= passed exclusion_dur, end=start of epoch) ')
                exclude_start=int(round((t_up_conv[ii]-exclusion_dur)*exclusion_sf))
                exclude_end=t_up_conv[ii]*exclusion_sf
                
            else:
                raise Exception ('Error: exclusion signal is passed in but no associated duration(s)')
            
            exclusion_trace=exclusion_signal[exclude_start:exclude_end]
    
            
            #determine if there are any upwards threshold crossing in the trace.
            exclusion_ts_above=last_pos_below_thresh(exclusion_trace,threshold=exclusion_thresh)
            
            #to account for cases where the exclusion trace starts in an "UP"-threshold state, look for downward going events
            exclusion_ts_below=last_pos_above_low_thresh(exclusion_trace,threshold=exclusion_thresh,low_threshold=False)
            
            if  exclusion_logic=='exclude' and (exclusion_ts_above.size==0) and (exclusion_ts_below.size==0):
                threshed_epochs_u.append(t_up_conv[ii])  
                threshed_epochs_d.append(t_down_conv[ii])
            
            elif exclusion_logic=='include' and exclusion_ts_above.size>0 :
                threshed_epochs_u.append(t_up_conv[ii])  
                threshed_epochs_d.append(t_down_conv[ii])

        epoch_o=zip(threshed_epochs_u,threshed_epochs_d)
        
    #to handle the last instance, append a very large value beyond where data was acquired
    epoch_o.append([trace_end+5,(trace_end+5.01)])
        
    epoch_start=[]
    epoch_end=[]
    
    #flag indicates that you are in a period which should be continued to the next step
    flag=0

    for i in range(len(epoch_o)-1):
        #get current epoch's start and end
        curr_s=epoch_o[i][0]
        curr_e=epoch_o[i][1]
        #get next epoch's start
        next_s=epoch_o[i+1][0]
            
        gap=next_s-curr_e
    
        if flag==0 and gap <=min_low:
            epoch_start.append(curr_s)
            flag=1
    
        elif flag==0 and gap>min_low:
            epoch_start.append(curr_s)
            epoch_end.append(curr_e)
        
        elif flag==1 and gap<=min_low:
            pass
    
        elif flag==1 and gap>min_low:
            epoch_end.append(curr_e)
            flag=0
        
        else:
            print ('error: could not be calculated')

    #measure duration of each period, exclude durations shorter than teh minimum time
    periods=[]
    for x in range(len(epoch_start)):
        
        duration=epoch_end[x]-epoch_start[x]
        if duration >= min_time:
            if isinstance(min_time_between, (int, float)) and min_time_between!=0.:
                #dtermine the time between this start period and the next
                if (x+1) >= len(epoch_start):
                    continue
                else:
                    time_between_next=epoch_start[x+1]-epoch_start[x]
                    if time_between_next>=min_time_between:
                            periods.append([epoch_start[x], epoch_end[x], duration])
                            
            else:
                periods.append([epoch_start[x], epoch_end[x], duration])
            
    
        
        
    
    periods=np.array(periods)
    
    if periods.size==0:
        print (' No periods found!!!')
    else:
        
        print ('Number of thresholded epochs found =  ' + str(np.shape(periods)[0]) + '. Median length (seconds) of each epoch is ' + str(np.median(periods[:,2])))
    #return start and end times + duration as array
    return periods
    

def signal_to_mean_noise (traces, sample_freq, signal_range, noise_range):
    '''calculate the signal to noise by sampling frequency power spectra
    :param traces: signal traces to calculate SNR from
    :param sample_freq: the acquisition sampling rate for the traces
    :param signal_range: the valid frequencies that define real signal (not noise), in Hz. List [x,x]
    :param noise_range: the frequency range that defines noise, in Hz. Average is caclulated across this. List [x,x]
    
    :Return log10 of the signal-to-noise for each trace
    
    '''
    import math
    
    snr=[]
    
    for xx in range(len(traces)):
        power_spect=np.abs(np.fft.fft(traces[xx], axis=0))**2
        scaling=(sample_freq+1)*20
        signal=np.max(power_spect[int(signal_range[0]*scaling):int(signal_range[1]*scaling)])
        noise=np.mean(power_spect[int(noise_range[0]*scaling):int(noise_range[1]*scaling)])
        snr.append(math.log10(signal/noise))
        
    return snr

 

def get_STA_traces_stamps(trace, frame_ts, event_onset_times, chunk_start, chunk_dur, dff=False, mean_frame_calculation='mean', verbose=True):
    '''
    Get stimulus-triggered average trace with input being in terms of time-stamps (digital) and being calculated around a stimulus of constant, known time. (constant periods, digital)
    Returns a stimulus-triggered averaged trace (STA trace) centered around a stimulus of set time with user defined pre- and post- stimulus periods.

    NOTE: units are important here. The trace, frame_ts, and event_onset_times must be in the same time base (likely seconds)
    
    :param trace: full trace that you wish to parse into a stimulus-triggered average (typically fluorescence trace)
    :param frame_ts: time stamps for when each sample from the trace (above) was captured. Typically this is imaging (2-P) frames.
    :param eventOnsetTimes: the timestamps for the stimulus/event of interest
    :param chunkStart: chunk start relative to the stimulus/event of interest
    :param chunkDur: duration of each chunk from the beginning of chunkstart, typically abs(pre_gap_dur)+post_gap_dur+stim_dur
    :param dff: whether to return values in terms of change from baseline normalized values (df/f)
    :param mean_frame_calculation: when calculating the mean duration of each of the frames (from Frame_TS) whether to use mean or median.
    :return: averaged trace of all chunks
    '''
    if mean_frame_calculation=='mean':
        mean_frame_dur = np.mean(np.diff(frame_ts))
        
    elif mean_frame_calculation=='median':
        mean_frame_dur = np.median(np.diff(frame_ts))
        
        
    chunk_frame_dur = int(np.ceil(chunk_dur / mean_frame_dur))
       
    if verbose:
        print ('Period Duration:'+ str(chunk_dur))
        print ('Mean Duration of Each Frame:' + str(mean_frame_dur))
        print ('Number of frames per period:' + str(chunk_frame_dur))
            
    chunk_start_frame = int(np.floor(chunk_start/ mean_frame_dur))
    
    traces = []
    df_traces=[]
    n = 0
    
    
    if np.shape(event_onset_times):
        
        for x in range(len(event_onset_times)):
            onset_frame_ind= np.argmin(np.abs(frame_ts-event_onset_times[x]))
            chunk_start_frame_ind = onset_frame_ind + chunk_start_frame
            chunk_end_frame_ind = chunk_start_frame_ind + chunk_frame_dur
                    
            if verbose:
                print ('Period:' +str(int(n)) + ' Starting frame index:' + str(chunk_start_frame_ind) + ' Ending frame index '+ str(chunk_end_frame_ind))
            
            if chunk_end_frame_ind <= trace.shape[0]:
                
                curr_trace = trace[chunk_start_frame_ind:chunk_end_frame_ind].astype(np.float32)
                traces.append(curr_trace)
                
                            
                df_curr_trace=stim_df_f(curr_trace, baseline_period=chunk_start, frame_rate=30)
                df_traces.append(df_curr_trace)
                
                n += 1
            else:
                print ('trace length ' + str(trace.shape[0]) + ' is shorter than the referenced start time for the next stimulus ' + str(chunk_end_frame_ind))
                continue
    
    else:
        print ('Only single stimulus period found, no averaging performed')
        onset_frame_ind= np.argmin(np.abs(frame_ts-event_onset_times))
        chunk_start_frame_ind = onset_frame_ind + chunk_start_frame
        chunk_end_frame_ind = chunk_start_frame_ind + chunk_frame_dur
        
        curr_trace=trace[chunk_start_frame_ind:chunk_end_frame_ind].astype(np.float32)
        traces = curr_trace
        df_traces=stim_df_f(curr_trace, baseline_period=chunk_start, frame_rate=30)
        
    if dff==True:
        return df_traces
    elif dff==False:
        return traces     



def get_event_trig_avg_stamps(trace, time_stamps, event_onset_times, event_end_times, time_before=1, time_after=1, verbose=True):
    '''
    Get event-triggered average trace with input being in terms of time-stamps (digital) and with events being irregular and NOT of constant time. (variable periods, digital)

    NOTE: units are important here. The trace, time_stamps, and event_onset_times must be in the same time base (likely seconds)
    
    :param trace: full trace that you wish to parse into a stimulus-triggered average (typically fluorescence trace)
    :param time_stamps: time stamps for when each sample from the trace (above) was captured. Typically this is imaging (2-P) frames.
    :param event_onset_times: the timestamps for the onset of the event of interest (units of time)
    :param event_end_times: the timestamps for the end of the event of interest (units of time)
    :param time_before: how much time before the event onset to include in the sample (units of time)
    :param time_after: how much time after the event onset to include in the sample (units of time)
    :return: traces for each onset and end, optionally with time before and after if specified by time_before and time_after
    '''
    traces = []
    n = 0

    for x in range(len(event_onset_times)):
        onset_frame_ind= np.argmin(np.abs(time_stamps-(event_onset_times[x]-time_before)))
        end_frame_ind=np.argmin(np.abs(time_stamps-(event_end_times[x]+time_after)))
        #chunk_start_frame_ind = onset_frame_ind + chunk_start_frame
        
        if verbose:
            print ('Chunk:' + str(int(n)) + ' Starting frame index:' + str(onset_frame_ind) + '; Ending frame index' + str(end_frame_ind))
        
        if end_frame_ind <= trace.shape[0]:
            
            curr_trace = trace[onset_frame_ind:end_frame_ind].astype(np.float32)
            traces.append(curr_trace)
            n += 1
        else:
            print ('trace length'+ str(trace.shape[0]) + ' is shorter than the referenced start time for the next stimulus ' +str(end_frame_ind))
            continue
    return traces



def get_STA_traces_samples (trace, event_onset_times, chunk_start, total_dur, sample_freq, dff=True, verbose=True):
    '''
    Get stimulus-triggered average(STA) trace for data where the units are not already in terms of seconds, but are in terms of continous samples (analog). (constant periods, analog)
    In this instance, traces are caclulated around a stimulus of constant, known time.

        
    :param trace: full trace that you wish to parse into a stimulus-triggered average (typically fluorescence trace). In this instance, it needs to be in terms of samples 
    :param event_onset_times: the timestamps for the stimulus/event of interest. These should be in units of time.
    :param chunk_start: chunk start relative to the stimulus/event of interest
    :param total_dur: duration of each chunk from the beginning of chunkstart, typically abs(pre_gap_dur)+post_gap_dur+stim_dur
    :return: traces for each period in units of the orginal sample units 
    '''

    #get the time in (in seconds) for each sample. This assume a linear spacing of samples.
    #total time (in seconds) for the trace
    sample_times=np.linspace(start=0,stop=trace.size/sample_freq,num=trace.size)
    
    #get t
    mean_sample_dur=np.mean(np.diff(sample_times))
    
    chunk_start= int(np.floor(chunk_start / mean_sample_dur))
    
    chunk_dur= int(np.ceil(total_dur/mean_sample_dur))
        
    if verbose:
        
        print ('# samples per period:' + str(chunk_dur))
        print ('mean duration of each sample (seconds): ' + str( mean_sample_dur))
    
    traces = []
    n = 0

    for x in range(len(event_onset_times)):
        
        #find the nearest analog sample to the event onset times
        onset_index=np.argmin(np.abs(sample_times-event_onset_times[x]))
        start_trace=onset_index + chunk_start
               
        end_trace=start_trace+chunk_dur
        
        if verbose:
            print ('Period:' + str(int(n)) + ' Starting frame index: ' + str(start_trace) + '. Ending frame index '+ str(end_trace))
        
        if end_trace <= trace.shape[0]:
            
            curr_trace = trace[start_trace:end_trace].astype(np.float32)
            traces.append(curr_trace)
            n += 1
            
        else:
            print ('trace length'+ str(trace.shape[0]) + ' is shorter than the referenced start time for the next stimulus ' +str(event_onset_times[x]))
            continue
        
    return traces          

def get_event_trig_avg_samples (trace, event_onset_times, event_end_times, sample_freq, time_before=2., time_after=2., dff=False,dff_baseline=0.,verbose=False):
    
    '''
    Get stimulus-triggered average(STA) trace for data where the units are not already in terms of seconds, but are in terms of continous samples (analog). (variable periods, analog)
    In this instance, traces are caclulated around a stimulus of constant, known time.

        
    :param trace: full trace that you wish to parse into a stimulus-triggered average (typically fluorescence trace). In this instance, it needs to be in terms of samples 
    :param event_onset_times: the timestamps for the stimulus/event of interest. These should be in units of time.
    :param event_end_times: the timestamps for the end of the stimulus/event of interest. 
    :param time_before: amount of extra time before the event onset to also extract
    :param time_after: amount of extra time AFTER the event onset to also extract
    :param dff: whether to return the resulting trace normalized to the time before period
    :param dff_baseline: allows for the specification of the df_f baseline in terms of relative time of the trace. If not specified the time_before variable is used.


    :return: traces for each period in units of the orginal sample units 
    '''
    #get the time in (in seconds) for each sample. This assume a constitent spacing of samples (i.e every 60 seconds).
    #total time (in seconds) for the trace
    


    sample_times=np.linspace(start=0,stop=trace.size/sample_freq,num=trace.size)
    
    traces = []
    n = 0

    if dff_baseline==0.:
        dff_baseline=time_before

    #check to see if there is multiple onset times (list) or if its a single instance

    if type(event_onset_times)==list:

        for x in range(len(event_onset_times)):
        
            #find the nearest analog sample to the event onset times
            start_trace=np.argmin(np.abs(sample_times-(event_onset_times[x]-abs(time_before))))
            end_trace=np.argmin(np.abs(sample_times-(event_end_times[x]+time_after)))
                   
            if verbose:
                print ('Period:' + str(int(n)) + ' Starting frame index: ' + str(start_trace) + '. Ending frame index '+ str(end_trace))
            
            if end_trace <= trace.shape[0]:

                if dff==True:

                    curr_trace=stim_df_f(arr=trace[start_trace:end_trace].astype(np.float32),baseline_period=dff_baseline,frame_rate=30.0)
                
                elif dff==False:
                    curr_trace = trace[start_trace:end_trace].astype(np.float32)
                
                traces.append(curr_trace)
                n += 1
            else:
                print ('trace length'+ str(trace.shape[0]) + ' is shorter than the referenced start time for the next stimulus ' )
                continue

    #take into account times where only a single onset is passed
    else:
        expected_dur=(event_end_times-event_onset_times)+time_before+time_after
        
        start_trace=np.argmin(np.abs(sample_times-(event_onset_times-abs(time_before))))
        end_trace=np.argmin(np.abs(sample_times-(event_end_times+time_after)))
        
        trace_dur=(end_trace-start_trace)/sample_freq
        
        if int(round(trace_dur))!=int(round(expected_dur)):
            if verbose:
                print ('Expected duration: ' + str(expected_dur) + ' and trace duration '+ str(trace_dur) + ' do not match ')
        
        if verbose:
                print ('Only single onset/end passed: Starting frame index:' + str(start_trace) + '.  Ending frame index: ' + str(end_trace) + '. Total dur ' + str(trace_dur))
        
        if end_trace <= trace.shape[0]:
            if dff==True:
                if verbose:
                    print ('Calculating DF/F from ')
                    print (dff_baseline)
                curr_trace=stim_df_f(arr=trace[start_trace:end_trace].astype(np.float32),baseline_period=dff_baseline,frame_rate=30.0)

            elif dff==False:
                curr_trace = trace[start_trace:end_trace].astype(np.float32)
        else:
            print ('trace length ' + str(trace.shape[0]) + ' is shorter than the referenced start time for the next stimulus')

        traces=curr_trace

    return traces


    
    

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

    from scipy.signal import convolve, boxcar

    #get the standard deviation across user-defined number of samples
    step=int(stdev_samples)
    stdev=[]
    for ii in range(0,microphone.shape[0],step):
        chunk=microphone[ii:ii+step]
        stdev.append(np.std(chunk))

    
    stdev_filtered=convolve(stdev, boxcar(M=filter_width))

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
    


def downsample_by_target_TS(signal, target_ts, signal_sf,verbose=True):
    '''
    Downsamples a signal to the same sampling frequency as a target signal. Accurate, slow method that can handle non-continous spacing between time-stamps.
    Averages by taking averaging half the interval before and after relative to the target signal time stamps
    
    Example: an signal is downsampled to match the same acquistion frequency as the imaging, averaging each half the interval in before a frame and half after the frame to produce a data point at the frame's time 

    This function assumes a continous signal is provided (not time stamps)
    
    :param signal:the signal you wish to downsample. 
    :param signal_sf: the sampling frequency (sf) of the signal variable (in Hz)
    :param target_ts: the time stamps of the target, assumes these are in seconds. CAN BE unevenly-spaced/irregular.

    :return: the signal downsampled to the frequency of the timestamps (target_ts)
    '''

    #get target signal average sampling frequncy
    target_sf=np.mean(np.diff(target_ts))

    #create a time array that has each time value for the passed in signal (based on sf)    
    times= np.linspace(start=0,stop=signal.size/(signal_sf*1000), num=signal.size)

    ds_sig=[]
    last_ind_behind=0
    counter=0
    for stamp in target_ts:

        #get timestamp half of the targets sf ahead and behind. Convert to ms
        ahead=(stamp+(0.5*target_sf))/1000
        behind=(stamp-(0.5*target_sf))/1000

        #find the nearest sample time for the signal relative to the time stamp

        ind_ahead=np.argmin(np.abs(times[last_ind_behind::]-ahead))
        
        ind_behind=np.argmin(np.abs(times[last_ind_behind::]-behind))
        
        avged_sample=np.mean(signal[ind_behind:ind_ahead],axis=0)

        #to speeed up, remember the last index
        last_ind_behind=ind_behind
        
        counter+=1
       
        if verbose==True:
            if counter%500==0 or counter==1:
                per=int(counter/float(len(target_ts))*100)
                print ('sample '+ str(counter)+ ' of '+ str(len (target_ts)) + ' processed.  '+ str( per ) + ' percent complete.')
        
        ds_sig.append(avged_sample)


    return np.array(ds_sig)


def fast_downsample_by_target_TS(signal, target_ts, signal_sf):
    '''Downsamples a signal to the same sampling frequency as a target signal in a fast manner. ANALOG
    
    ***Assumes the target_ts for sampling must be regularly spaced and therefore does not drift*** 
        
    Averages by taking averaging half the interval before and after relative to the target signal time stamps
    
    Example: an signal is downsampled to match the same acquistion frequency as the imaging, averaging each half the interval in before a frame and half after the frame to produce a data point at the frame's time 

    This function assumes 
    
    :param signal:the signal you wish to downsample. A continous signal (not time stamps, ANALOG).
    :param target_ts: the time stamps of the target, assumes these are in seconds. CANNOT BE UNEVENLY SPACED
    :param signal_sf: the sampling frequency (sf) of the signal variable (in Hz)


    :return: the signal downsampled to the frequency of 
    '''

    #create a time array that has each time value for the passed in signal (based on sf)    
    times= np.linspace(start=0,stop=signal.size/signal_sf,num=signal.size)

    #first calculate the offset for when the time-stamped signal begins relative to the signal
    start_offset=np.argmin(np.abs(times-target_ts[0]))

    #calculate the number of samples between each timestamp
    ts_two=np.argmin(np.abs(times-target_ts[1]))
    ts_three=np.argmin(np.abs(times-target_ts[2]))
    sample_space=ts_three-ts_two

    resampled_loop=[]

    avg_samp=int(sample_space*0.25)

    for xx in range(len (target_ts)):
        #calculate 0.25 of the interval before and after
    
        #determine the nearest signal sample to the target ts
        center=start_offset+sample_space*xx
        #determine how many samples before and after for averaging (based on avg_samp)
        before=center-avg_samp
        after=center+avg_samp
    
        resampled=np.nanmean(signal[before:after],axis=0)
    
        resampled_loop.append(resampled)

    return np.array(resampled_loop)


def downsample_TS_by_target_TS(signal, signal_ts, target_ts, time_around=0.0,verbose=True):
    '''
    Downsamples a signal to the same sampling frequency as a target signal. Accurate, slow method that can handle non-continous spacing between time-stamps.
    Averages by taking averaging half the interval before and after relative to the target signal time stamps
    
    Example: an signal is downsampled to match the same acquistion frequency as the imaging, averaging each half the interval in before a frame and half after the frame to produce a data point at the frame's time 

    This function assumes time stamps, not a continous signal, is provided (DIGITAL)
    
    :param signal:the signal t-series/trace you wish to downsample. Should be digital and orginate with timestamps (target_ts) 
    :param signal_ts: time stamps of the signal on the same time base as the target, assumes these are in seconds. 
    :param target_ts: the time stamps of the target, assumes these are in seconds. CAN BE unevenly-spaced/irregular.
    :param time_around: amount of time in fraction of target_ts to average before and after to create a 'downsample'
                        Example: time_around= 0.5; for a 30 hz target SF, each sample is collected at 33 ms. 
                        time around in this instance is 16.5 ms. Therefore, for the nearest signal_ts to a given target_ts,
                        any signal data 49.5 ms before and 49.5 ms after is averaged in to produce a single time point.

    :return: the signal trace downsampled to the frequency of the timestamps (target_ts)
    '''

    #calculate the sampling rate of the signal
    sf=1/np.mean(np.diff(signal_ts))
    
    #get target signal average sampling frequncy
    target_sf=np.mean(np.diff(target_ts))
    #create a time array that has each time value for the passed in signal (based on sf)    
    times= np.linspace(start=0,stop=(signal_ts.size/sf),num=signal_ts.size)

    ds_sig=[]
    counter=0
    for stamp in target_ts:

        #get timestamp half of the targets sf ahead and behind. Convert to ms

        #find the nearest sample time for the signal relative to the time stamp

        if time_around>0:
                        
            span=time_around*target_sf
            
            ahead=(stamp+span)
                        
            behind=(stamp-span)

            ind_ahead=np.argmin(np.abs(times-ahead))

            ind_behind=np.argmin(np.abs(times-behind))
        
            avged_sample=np.nanmean(signal[ind_behind:ind_ahead],axis=0)
                   
        else:
            ind=np.argmin(np.abs(times-stamp))
           
            avged_sample=signal[ind]
      
        counter+=1
       
        if verbose==True:
            if counter%500==0 or counter==1:
                                
                per=int(counter/float(len(target_ts))*100)

                print ('sample '+ str(counter)+ ' of '+ str(len (target_ts)) + ' processed.  '+ str( per ) + ' percent complete.')
        
        ds_sig.append(avged_sample)
        
    return np.array(ds_sig)



def stimulus_thresh_df (paths,data_key, thresh_signal, thresh, min_l, min_t, 
                        before, after, baseline_period,response_period,min_time_between=False,use_dff=True,
                        other_signals=[],dff_baseline_dur=1., exclusion_sig='null',exclusion_thresh=0.,exclusion_dur=0.,exclusion_logic='exclude',
                        override_ends=False, use_per_thresh=False, sample_freq=30. ):
    
    """
    :param paths: path to HDF5 files
    :param data_key: key for the HDF5 to access the data type of interest
    
    ---Thresholding parameters
    
    :param thresh_signal: the signal to threshold on
    :param thresh: the threshold
    :param min_l: the minimum amount of time the signal must go below the threshold to end a period
    :param min_t: minimum time for a threshold period
    :param min_time_between: the minimum amount that must be between the start of two epochs. Useful for finding epochs that occur in isolation from nearby other epochs.
    
    ---trace extraction parameters
    
    :param before: amount of time before the threshold time to extract
    :param after: amount of time after the threshold time to extract
    :param baseline: how many seconds in the the 'before' period to calculate baseline periods from (used in DF/F calculations and others)
    :param baseline: where the "baseline" should be calculated from in the trace (used in DF/F calculations and others) . Tuple of start time and end time for the baseline.
    :param sample_t_after_thresh: when sampling the "response" start this far after the threshold crossing (0 = at the threshold). Set to string 'half' to sample 50% through the epoch's duration.
    :param sample_dur_after_thresh:when sampling the "response" start from sample_t_after_thresh and go this many seconds ahead
    
    """
    import os
    import h5py
    import pandas as pd
    
    #create dataframe of all ROI responses for every running epoch

    total_roi_counter=0
    responses=[]
    meaned_responses=[]
    
    #check to make sure that the baseline is specified as a tuple and deal with instances where it isn't
    if isinstance(baseline_period, (int, float)):
        print ('the baseline period was specified as a single number, not a start and end time. Assuming start time is time 0 and end time of hte baseline is what is specified.')
        baseline_period=(0,baseline_period)
    

    for path in paths:
        mouse_id=os.path.basename(path)[0:7]
        print ('\n processing ' + str(mouse_id) + '\n')
    
        data_f=h5py.File(path,'r')
        data=data_f.get(data_key)
        
        if use_per_thresh==True:
            #first lowpass filter and calculate the median of the trace
            median=np.nanmedian(butter_lowpass_filter(data[thresh_signal], cutoff=1., analog=True))
            threshold_per=median+(thresh*median)
            thresh=threshold_per
            
        if exclusion_sig=='null':
            runs=threshold_period(signal=data[thresh_signal], threshold=thresh,
                                  min_low=min_l, sample_freq=30., min_time=min_t)
            
        else:
            print (exclusion_logic+'  epochs where the '+ str(exclusion_sig) + ' is greater than '+ str(exclusion_thresh))
            
            runs=threshold_period(signal=data[thresh_signal], threshold=thresh,min_time_between=min_time_between,
                                  min_low=min_l, sample_freq=30., min_time=min_t,exclusion_signal=data[exclusion_sig],
                                  exclusion_dur=exclusion_dur,exclusion_logic=exclusion_logic,
                                  exclusion_thresh=exclusion_thresh)

        #check if no threshold crossing are found. If so, go to next file
        if runs.size==0:
            print (' No periods found for id '+ str(mouse_id))
            continue
        
        #get teh start times from teh threshold_period output
        starts=runs[:,0]
        #take into account times where you want to get traces that start relative to the onset and you don't want to be concerned with their duration
        if override_ends==False:
            starts=runs[:,0]
            ends=runs[:,1]
            durs=runs[:,2]
    
        elif isinstance(override_ends, (int, float)):
           #if a number is passed to override the ends, determine the end of the periods by adding this number to the beginning
           print ('Overiding detected durations and using USER-DEFINED durations')
           starts=runs[:,0]
           ends=starts+override_ends
           durs=ends-starts
    
        elif override_ends=='starts': 
            print ('setting the start times equal to the detected END TIMES!')
            starts=runs[:,1]
            ends=runs[:,1]
            durs=(ends-starts)+1.
        
        error_counter=0
        #calculate the stimulus evoked dff for each roi
        #loop for each ROI
        for roi in range(len(data['axon_traces'])):
        
            mean_onset_list=[]
            mean_end_list=[]
            mean_speed_list=[]
            mean_delta_speed_list=[]
            
            #create a list to store the first portion of each trace where there always a epoch peroiod
            traces_onset=[]
            #create a more inclusive list to store entire baseline, onset, and after periods for arbituarily selecting regions for analysis
            before_after_traces=[]
            
            #determine unique ids for each roi and calculate area 
            roi_unique_id=mouse_id[-6::]+'_'+ str(0)+str(roi)
            mask=data_f['masks']['axon_masks'][roi]
            pixels=np.where(np.logical_and(mask!=0, ~np.isnan(mask)))
            roi_area=np.shape(pixels)[0]*np.shape(pixels)[1]
            
            #loop for each epoch
            for xx in range(len(starts)):
                
                runnings=get_event_trig_avg_samples(data[thresh_signal],event_onset_times=starts[xx],
                              event_end_times=ends[xx],
                              sample_freq=sample_freq,
                              time_before=before, 
                              time_after=after, verbose=False)
                
                
                if response_period[1]=='half':
                    if override_ends==False:
                        response_period_end=response_period[0]+durs[xx]/2.
                    elif isinstance(override_ends, (int, float)):
#                            print ('Half duration is passed for end, but overriding durations: calculating duration from half the time of after')
                        response_period_end=response_period[0]+(after/2.)
                            
                    else:
                        print ('major error')
                else:
                    response_period_end=response_period[1]
                        
            
                baseline_indices=(int((baseline_period[0]*sample_freq)), int((baseline_period[1]*sample_freq)))
                response_indices=(int((response_period[0]*sample_freq)), int((response_period_end*sample_freq)))
                
                
                #get mean running_speed
                baseline_speed=np.nanmean(runnings[baseline_indices[0]:baseline_indices[1]],axis=0)
                mean_speed=np.nanmean(runnings[response_indices[0]:response_indices[1]],axis=0)
                delta_speed=mean_speed-baseline_speed
   
                #produce an array that is composed of each ROI's DF/F epeoch
                
                axon_responses=get_event_trig_avg_samples(data['axon_traces'][roi],event_onset_times=starts[xx],
                              event_end_times=ends[xx],
                              sample_freq=sample_freq,
                              time_before=before, 
                              time_after=after, dff=use_dff,dff_baseline=(baseline_period[0], baseline_period[1]), verbose=False)
                
                #check to make sure expected durations match returned trace durations
                expected_dur=((ends[xx]-starts[xx])+before+after)
                trace_dur_run=int(round(len(runnings)/30.))
                trace_dur_axon=int(round(len(axon_responses)/30.))
 
                
                dur_check=int( round( (ends[xx]-starts[xx]+before+after)*30.))
                if len(axon_responses)!=dur_check:
                    if error_counter==0:
                        print ('Epoch # ' + str(xx) + ' Trace durations do not match expected duration: Likely due to not enough samples to grab. Skipping')
                        error_counter+=1
                    continue
                
                
                if ((trace_dur_run!=int(round(expected_dur))) or (trace_dur_axon!= int(round(expected_dur))) ) :
                    if error_counter==0:
                        
                        print ('Epoch # ' + str(xx) +'. Epoch length mismatch warning: Expected duration: ' + str(int(expected_dur)) + ' and trace duration '+ str(int(trace_dur_run)) + ' do not match ')
                        print ('skipping event/epoch')
                        error_counter+=1
                    continue
                
                #get any other signals the user may want
                others=[]
                others_means=[]
                
                for extras in other_signals:
                    #get the associated running trace
                    sig=get_event_trig_avg_samples(data[extras],event_onset_times=starts[xx],
                              event_end_times=ends[xx],
                              sample_freq=sample_freq,
                              time_before=before, 
                              time_after=after, verbose=False)
                    
                    baseline_sig=np.nanmean(sig[baseline_indices[0]:baseline_indices[1]],axis=0)
                    mean_sig=np.nanmean(sig[response_indices[0]:response_indices[1]],axis=0)
                    #calculate in terms of percent change of baseline
                    delta_sig=(mean_sig-baseline_sig)/baseline_sig*100
                    
                    onset_sig=sig[int(before*sample_freq)+1]
                
                    others.append(sig)
                    others_means.append([baseline_sig, onset_sig, mean_sig, delta_sig])
                
                #calculate a trace that MUST include the region betweeen start and end. This is performed to allow for averaging of the epochs that have different durations. 
                #it always will produce a trace that contains the MINIMAL length resonse
                end_of_eval_period_for_sig= int(round(((before+min_t)*sample_freq)))
                onset_trace=axon_responses[0:end_of_eval_period_for_sig+1]
                traces_onset.append(onset_trace)
                #calculate a trace that includes the baseline period, the onset, and the amount of time after. used in calculation of signficance for an ROI
                before_after_trace=axon_responses[0:int((before+after)*sample_freq)]
                before_after_traces.append(before_after_trace)
               
                
                #get the DF at the threshold crossing
                onset_df=axon_responses[int(before*sample_freq)+1]
                #end_index=int(ends[xx]*sample_freq)
                
                end_index=int((before*sample_freq)+(durs[xx]*sample_freq)-1)
                end_df=axon_responses[end_index]

        
                mean_df=np.nanmean(axon_responses[response_indices[0]:response_indices[1]],axis=0)
    
                #append to list: roi number, mouse_id, epoch number,
                #start_time, end_time, duration, axon response array (DF),
                #mean df_f responses at user-define time, running array, mean_speed
                sublist=[roi_unique_id,mouse_id, xx, starts[xx],ends[xx],durs[xx],
                         axon_responses, onset_df, mean_df,end_df,
                         runnings,mean_speed,delta_speed,roi_area,total_roi_counter]
                
                for yy in range(len(others)):
                    sublist.append(others[yy])
                    #baseline_sig
                    sublist.append(others_means[yy][0])
                    #peak_sig
                    sublist.append(others_means[yy][1])
                    #mean_sig
                    sublist.append(others_means[yy][2])
                    #delta_sig
                    sublist.append(others_means[yy][2])
                
                responses.append(sublist)

                mean_onset_list.append(onset_df)
                mean_end_list.append(end_df)
                mean_speed_list.append(mean_speed)
                mean_delta_speed_list.append(delta_speed)
                
            #get the mean trace from the onset and beginning of thresholded region
            mean_onset_trace=np.nanmean(traces_onset,axis=0)
            #determine if the average response for the ROI is significant
            
            #12_6 change: allow significance to be calculated from arbituary regions across the entire baselein and end period, not just the consistently resposne
            #therefore use the onset plus and minus the 
            before_after_mean=np.nanmean(before_after_traces,axis=0)
            
            pvalue=significant_response(before_after_mean, base_period=(baseline_period[0],baseline_period[1]), stim_period=(response_period[0],response_period_end), sample_freq=30.)    
            if pvalue < 0.05:
                significant=True
            else:
                significant=False
            
      
            mean_onset_df_roi=np.nanmean(np.asarray(mean_onset_list),axis=0)
            mean_end_df_roi=np.nanmean(np.asarray(mean_end_list), axis=0)
            mean_speed_roi=np.nanmean(np.asarray(mean_speed_list),axis=0)
            mean_delta_speed_roi=np.nanmean(np.asarray(mean_delta_speed_list),axis=0)
        
            meaned_responses.append([roi_unique_id, mouse_id,pvalue,significant, mean_onset_df_roi,mean_end_df_roi,
                                 mean_speed_roi,mean_delta_speed_roi,total_roi_counter,before_after_mean,mean_onset_trace])
    
            total_roi_counter+=1
        
    column_names=['roi id','mouse_ID','epoch number', 'start time', 'end time', 'duration',
                                   'axon trace', 'onset df', 'peak df', 'end df',
                                   'threshold signal trace', 'peak thresh value', 'delta of thresh trace', 'ROI area','roi number']
    for names in other_signals:
        column_names.append(names)
        column_names.append(str(names) + ' baseline sig')
        column_names.append(str(names) + ' onset sig')
        column_names.append(str(names) + ' peak sig')
        column_names.append(str(names) + ' delta % sig')
        
        

        
    df=pd.DataFrame(responses,columns=column_names)
    
    df_mean=pd.DataFrame(meaned_responses,columns=['roi id','mouse_ID','p value', 'significant mean resp', 'mean onset df', 'mean end df',
                                               'mean thresh signal', 'mean delta thresh signal', 'roi number','mean trace', 'mean baseline and onset trace'])

    #add whether the mean response is significant to the df mean
    
    mean_sigs=[]
    for index, row in df.iterrows():
        roi_num=df['roi number'][index]
    
        #get whether it is significant on average
        mean_p=float(df_mean.loc[(df_mean['roi number']==roi_num)]['p value'])
    
        if mean_p < 0.05:
            significant=True
        else:
            significant=False
        
        mean_sigs.append([mean_p, bool(significant)])

    df_sig_responses=pd.DataFrame(mean_sigs, columns=['mean p value', 'mean sig'])

    df=pd.concat([df,df_sig_responses], axis=1)
    
    #clean up dataframes by re-indexing by the roi_ids
    df=df.sort_values(['roi id', 'epoch number'])
    df.reset_index(drop=True)
    
    df_mean=df_mean.sort_values(['roi id'])
    df_mean.reset_index(drop=True)
    
        
        
    
    return df,df_mean






def create_df_from_timestamps(data,mouse_id,masks,starts,ends,baseline_period,response_period, before,after, min_t=0.5,
                              override_ends=False,thresh_signal=False,other_signals=[],sample_freq=30.,use_dff=True,total_roi_counter=0. ):
    '''
    :param paths: path to HDF5 files
    :param data_key: key for the HDF5 to access the data type of interest
    '''
    import pandas as pd
    
    responses=[]
    meaned_responses=[]
    error_counter=0
    #Get event triggered responses for each ROI
    for roi in range(len(data['axon_traces'])):
        
        
        mean_onset_list=[]
        mean_end_list=[]
        mean_speed_list=[]
        mean_delta_speed_list=[]
            
        #create a list to store the first portion of each trace where there always a epoch peroiod
        traces_onset=[]
        #create a more inclusive list to store entire baseline, onset, and after periods for arbituarily selecting regions for analysis
        before_after_traces=[]
        
        #determine unique ids for each roi and calculate area 
        roi_unique_id=mouse_id[-6::]+'_'+ str(0)+str(roi)
        mask=masks[roi]
        pixels=np.where(np.logical_and(mask!=0, ~np.isnan(mask)))
        roi_area=np.shape(pixels)[0]*np.shape(pixels)[1]
        
        #loop for each epoch/stimulus
        for xx in range(len(starts)):
            
        
            if response_period[1]=='half':
                #check to see if the ends should be ignored
                if override_ends==False:
                    response_period_end=response_period[0]+(ends[xx]-starts[xx])/2.
                    
                elif isinstance(override_ends, (int, float)):
                    response_period_end=response_period[0]+(after/2.)
                        
                else:
                    print ('major error')
            else:
                response_period_end=response_period[1]
                    
            baseline_indices=(int((baseline_period[0]*sample_freq)), int((baseline_period[1]*sample_freq)))
            response_indices=(int((response_period[0]*sample_freq)), int((response_period_end*sample_freq)))
            
            expected_dur=((ends[xx]-starts[xx])+before+after)
            dur_check=int( round( (ends[xx]-starts[xx]+before+after)*30.))
            
        #check to see if a thresholded signal is passed in. If so extract its parameters    
            if thresh_signal:
                if override_ends=='starts': 
                    runnings=get_event_trig_avg_samples(data[thresh_signal],event_onset_times=ends[xx], event_end_times=ends[xx]+1,
                                                    sample_freq=sample_freq,time_before=before, time_after=after, verbose=False)
                else:     
                    #get the signal trace that was thresholded
                    runnings=get_event_trig_avg_samples(data[thresh_signal],event_onset_times=starts[xx],
                                                            event_end_times=ends[xx],
                                                            sample_freq=sample_freq,
                                                            time_before=before, 
                                                            time_after=after, verbose=False)

            
                #get mean changes in the thresholded signal
                baseline_speed=np.nanmean(runnings[baseline_indices[0]:baseline_indices[1]],axis=0)
                mean_speed=np.nanmean(runnings[response_indices[0]:response_indices[1]],axis=0)
                delta_speed=mean_speed-baseline_speed
                
                trace_dur_run=int(round(len(runnings)/30.))
                
                if trace_dur_run!=int(round(expected_dur)):
                    print ('Epoch # ' + str(xx) +'. Epoch length mismatch warning: Expected duration: ' + str(int(expected_dur)) + ' and the THRESHOLDED Signal trace duration '+ str(int(trace_dur_run)) + ' do not match ')
                    print ('skipping event/epoch')
                    
                    continue
   
            #produce an array that is composed of each ROI's DF/F epeoch
            axon_responses=get_event_trig_avg_samples(data['axon_traces'][roi],event_onset_times=starts[xx],
                          event_end_times=ends[xx],
                          sample_freq=sample_freq,
                          time_before=before, 
                          time_after=after, dff=use_dff,dff_baseline=(baseline_period[0], baseline_period[1]), verbose=False)
            
            #check to make sure expected durations match returned trace durations
            trace_dur_axon=int(round(len(axon_responses)/30.))
 
            if len(axon_responses)!=dur_check:
                if error_counter==0:
                    print ('Epoch # ' + str(xx) + ' Trace durations do not match expected duration: Likely due to not enough samples to grab. Skipping')
                    error_counter+=1
                continue
            

            if trace_dur_axon!= int(round(expected_dur)):
                if error_counter==0:
                    print ('Epoch # ' + str(xx) +'. Epoch length mismatch warning: Expected duration: ' + str(int(expected_dur)) + ' and trace duration '+ str(int(trace_dur_run)) + ' do not match ')
                    print ('skipping event/epoch')
                    error_counter+=1
                continue
            
            #get any other signals the user may want
            others=[]
            others_means=[]
            
            for extras in other_signals:
                #get the associated running trace
                sig=get_event_trig_avg_samples(data[extras],event_onset_times=starts[xx],
                          event_end_times=ends[xx],
                          sample_freq=sample_freq,
                          time_before=before, 
                          time_after=after, verbose=False)
                
                baseline_sig=np.nanmean(sig[baseline_indices[0]:baseline_indices[1]],axis=0)
                mean_sig=np.nanmean(sig[response_indices[0]:response_indices[1]],axis=0)
                #calculate in terms of percent change of baseline
                delta_sig=(mean_sig-baseline_sig)/baseline_sig*100
                
                onset_sig=sig[int(before*sample_freq)+1]
            
                others.append(sig)
                others_means.append([baseline_sig, onset_sig, mean_sig, delta_sig])
            
            #calculate a trace that MUST include the region betweeen start and end. This is performed to allow for averaging of the epochs that have different durations. 
            #it always will produce a trace that contains the MINIMAL length resonse
            end_of_eval_period_for_sig= int(round(((before+min_t)*sample_freq)))
            onset_trace=axon_responses[0:end_of_eval_period_for_sig+1]
            traces_onset.append(onset_trace)
            #calculate a trace that includes the baseline period, the onset, and the amount of time after. used in calculation of signficance for an ROI
            before_after_trace=axon_responses[0:int((before+after)*sample_freq)]
            before_after_traces.append(before_after_trace)
           
            
            #get the DF at the threshold crossing
            onset_df=axon_responses[int(before*sample_freq)+1]
            #end_index=int(ends[xx]*sample_freq)
            
            end_index=int((before*sample_freq)+((ends[xx]-starts[xx])*sample_freq)-1)
            end_df=axon_responses[end_index]
            mean_df=np.nanmean(axon_responses[response_indices[0]:response_indices[1]],axis=0)

            #create a list that is the basis of the dataframe
            sublist=[roi_unique_id,mouse_id, xx, starts[xx],ends[xx],(ends[xx]-starts[xx]),
                     axon_responses, onset_df, mean_df,end_df,
                     roi_area,total_roi_counter]
            #create a list that will be columns corresponding to variabels above
            column_names=['roi id','mouse_ID','epoch number', 'start time', 'end time', 'duration',
                               'axon trace', 'onset df', 'peak df', 'end df',
                               'ROI area','roi number']
            
            
            
            if thresh_signal:
                sublist.append(runnings)
                sublist.append(mean_speed)
                sublist.append(delta_speed)
                
                column_names.append('threshold signal trace')
                column_names.append('mean_speed_roi')
                column_names.append('mean_delta_speed_roi')
            
            
            for yy in range(len(others)):
                sublist.append(others[yy])
                #baseline_sig
                sublist.append(others_means[yy][0])
                #peak_sig
                sublist.append(others_means[yy][1])
                #mean_sig
                sublist.append(others_means[yy][2])
                #delta_sig
                sublist.append(others_means[yy][2])
            
            responses.append(sublist)

            mean_onset_list.append(onset_df)
            mean_end_list.append(end_df)
            if thresh_signal:
                mean_speed_list.append(mean_speed)
                mean_delta_speed_list.append(delta_speed)
        
        
        #get the mean trace from the onset and beginning of thresholded region
        mean_onset_trace=np.nanmean(traces_onset,axis=0)
        #determine if the average response for the ROI is significant
        
        #12_6 change: allow significance to be calculated from arbituary regions across the entire baselein and end period, not just the consistently resposne
        #therefore use the onset plus and minus the 
        before_after_mean=np.nanmean(before_after_traces,axis=0)
        
        pvalue=significant_response(before_after_mean, base_period=(baseline_period[0],baseline_period[1]), stim_period=(response_period[0],response_period_end), sample_freq=30.)    
        if pvalue < 0.05:
            significant=True
        else:
            significant=False
        
  
        mean_onset_df_roi=np.nanmean(np.asarray(mean_onset_list),axis=0)
        mean_end_df_roi=np.nanmean(np.asarray(mean_end_list), axis=0)


        if thresh_signal:
            mean_speed_roi=np.nanmean(np.asarray(mean_speed_list),axis=0)
            mean_delta_speed_roi=np.nanmean(np.asarray(mean_delta_speed_list),axis=0)
                        
            meaned_responses.append([roi_unique_id, mouse_id,pvalue,significant, mean_onset_df_roi,mean_end_df_roi,
                             total_roi_counter,before_after_mean,mean_onset_trace,mean_speed_roi,mean_delta_speed_roi]) 
            
            meaned_columns=['roi id','mouse_ID','p value', 'significant mean resp', 'mean onset df', 'mean end df','roi number',
                        'mean trace', 'mean baseline and onset trace','mean thresh signal', 'mean delta thresh signal']
        else:
            meaned_responses.append([roi_unique_id, mouse_id,pvalue,significant, mean_onset_df_roi,mean_end_df_roi,
                             total_roi_counter,before_after_mean,mean_onset_trace]) 
            
            meaned_columns=['roi id','mouse_ID','p value', 'significant mean resp', 'mean onset df', 'mean end df','roi number',
                        'mean trace', 'mean baseline and onset trace']
        
 
    
        total_roi_counter+=1     
    
    for names in other_signals:
        column_names.append(names)
        column_names.append(str(names) + ' baseline sig')
        column_names.append(str(names) + ' onset sig')
        column_names.append(str(names) + ' peak sig')
        column_names.append(str(names) + ' delta % sig')
    
    
    df=pd.DataFrame(responses,columns=column_names)

    
    df_mean=pd.DataFrame(meaned_responses,columns=meaned_columns)

    #add whether the mean response is significant to the df mean

    mean_sigs=[]
    for index, row in df.iterrows():
        roi_num=df['roi id'][index]

        #get whether it is significant on average
        mean_p=float(df_mean.loc[(df_mean['roi id']==roi_num)]['p value'])

        if mean_p < 0.05:
            significant=True
        else:
            significant=False
    
        mean_sigs.append([mean_p, bool(significant)])

    df_sig_responses=pd.DataFrame(mean_sigs, columns=['mean p value', 'mean sig'])

    df=pd.concat([df,df_sig_responses], axis=1)

    #clean up dataframes by re-indexing by the roi_ids
    df=df.sort_values(['roi id', 'epoch number'])
    df.reset_index(drop=True)

    df_mean=df_mean.sort_values(['roi id'])
    df_mean.reset_index(drop=True)

    return df,df_mean,total_roi_counter


def new_stimulus_thresh_df (paths,data_key, thresh_signal, thresh, min_l, min_t, 
                        before, after, baseline_period,response_period,min_time_between=False,use_dff=True,
                        other_signals=[], exclusion_sig='null',exclusion_thresh=0.,exclusion_dur=0.,exclusion_logic='exclude',
                        override_ends=False, use_per_thresh=False, sample_freq=30. ):
    
    """
    :param paths: path to HDF5 files
    :param data_key: key for the HDF5 to access the data type of interest
    
    ---Thresholding parameters
    
    :param thresh_signal: the signal to threshold on
    :param thresh: the threshold
    :param min_l: the minimum amount of time the signal must go below the threshold to end a period
    :param min_t: minimum time for a threshold period
    :param min_time_between: the minimum amount that must be between the start of two epochs. Useful for finding epochs that occur in isolation from nearby other epochs.
    
    ---trace extraction parameters
    
    :param before: amount of time before the threshold time to extract
    :param after: amount of time after the threshold time to extract
    :param baseline: how many seconds in the the 'before' period to calculate baseline periods from (used in DF/F calculations and others)
    :param baseline: where the "baseline" should be calculated from in the trace (used in DF/F calculations and others) . Tuple of start time and end time for the baseline.
    :param sample_t_after_thresh: when sampling the "response" start this far after the threshold crossing (0 = at the threshold). Set to string 'half' to sample 50% through the epoch's duration.
    :param sample_dur_after_thresh:when sampling the "response" start from sample_t_after_thresh and go this many seconds ahead
    
    """
    
    #check to make sure that the baseline is specified as a tuple and deal with instances where it isn't
    if isinstance(baseline_period, (int, float)):
        print ('the baseline period was specified as a single number, not a start and end time. Assuming start time is time 0 and end time of hte baseline is what is specified.')
        baseline_period=(0,baseline_period)
    
    
    dfs=[]
    df_means=[]
    for path in paths:
        mouse_id=os.path.basename(path)[0:7]
        print ('\n processing ' + str(mouse_id) + '\n')
    
        data_f=h5py.File(path,'r')
        data=data_f.get(data_key)
        masks=data_f['masks']['axon_masks']
        
        #performing threshold on each HDF5
        
        if use_per_thresh==True:
            #first lowpass filter and calculate the median of the trace
            median=np.nanmedian(butter_lowpass_filter(data[thresh_signal], cutoff=1., analog=True))
            threshold_per=median+(thresh*median)
            thresh=threshold_per
            
        if exclusion_sig=='null':
            runs=threshold_period(signal=data[thresh_signal], threshold=thresh,
                                  min_low=min_l, sample_freq=30., min_time=min_t)
            
        else:
            print (exclusion_logic+'  epochs where the '+ str(exclusion_sig) + ' is greater than '+ str(exclusion_thresh))
            
            runs=threshold_period(signal=data[thresh_signal], threshold=thresh,min_time_between=min_time_between,
                                  min_low=min_l, sample_freq=30., min_time=min_t,exclusion_signal=data[exclusion_sig],
                                  exclusion_dur=exclusion_dur,exclusion_logic=exclusion_logic,
                                  exclusion_thresh=exclusion_thresh)

        #check if no threshold crossing are found. If so, go to next file
        if runs.size==0:
            print (' No periods found for id '+ str(mouse_id))
            continue
        
        #get teh start times from teh threshold_period output
        starts=runs[:,0]
        #take into account times where you want to get traces that start relative to the onset and you don't want to be concerned with their duration
        if override_ends==False or override_ends=='starts':
            ends=runs[:,1]
            durs=runs[:,2]
    
        elif isinstance(override_ends, (int, float)):
           #if a number is passed to override the ends, determine the end of the periods by adding this number to the beginning
           print ('Overiding detected durations and using USER-DEFINED durations')
           ends=starts+override_ends
           durs=ends-starts
    
        df,df_mean=create_df_from_timestamps(data=data,masks=masks,starts=starts,ends=ends,baseline_period=baseline_period,response_period=response_period, before=before,after=after,mouse_id=mouse_id,min_t=min_t,
                                             other_signals=other_signals,override_ends=override_ends,thresh_signal=thresh_signal, use_dff=use_dff,sample_freq=30. )
       
        dfs.append(df)
        df_means.append(df_mean)

    
    return pd.concat(dfs), pd.concat(df_means)



def stim_df (paths, data_key, time_stamps_key, stim_dur ,before, after, 
             baseline_period,response_period,
             other_signals, use_dff=True, sample_freq=30. ):
    
    """
    :param paths: path to HDF5 files
    :param data_key: key for the HDF5 to access the data type of interest
    
    ---Thresholding parameters
    
    :param thresh_signal: the signal to threshold on
    :param thresh: the threshold
    :param min_l: the minimum amount of time the signal must go below the threshold to end a period
    :param min_t: minimum time for a threshold period
    :param min_time_between: the minimum amount that must be between the start of two epochs. Useful for finding epochs that occur in isolation from nearby other epochs.
    
    ---trace extraction parameters
    
    :param before: amount of time before the threshold time to extract
    :param after: amount of time after the threshold time to extract
    :param baseline: how many seconds in the the 'before' period to calculate baseline periods from (used in DF/F calculations and others)
    :param baseline: where the "baseline" should be calculated from in the trace (used in DF/F calculations and others) . Tuple of start time and end time for the baseline.
    :param sample_t_after_thresh: when sampling the "response" start this far after the threshold crossing (0 = at the threshold). Set to string 'half' to sample 50% through the epoch's duration.
    :param sample_dur_after_thresh:when sampling the "response" start from sample_t_after_thresh and go this many seconds ahead
    
    """
    
    #check to make sure that the baseline is specified as a tuple and deal with instances where it isn't
    if isinstance(baseline_period, (int,float)):
        print ('the baseline period was specified as a single number, not a start and end time. Assuming start time is time 0 and end time of hte baseline is what is specified.')
        baseline_period=(0,baseline_period)
    

    dfs=[]
    df_means=[]
    roi_counter=0
    for path in paths:
        mouse_id=os.path.basename(path)[0:7]
        print ('\n processing ' + str(mouse_id) + '\n')
    
        data_f=h5py.File(path,'r')
        data=data_f.get(data_key)
        masks=data_f['masks']['axon_masks']
        
        time_stamps=data.get(time_stamps_key)
        
        #check to see if the duration is given as an HDF5 key to the 
        if isinstance(stim_dur, str):
            stim_dur=data[time_stamps_key].attrs[stim_dur]
            print("Read in Stimulus Duration from HDF5. Reported Stimulation duration is : " + str(stim_dur) + " seconds \n")
            
        starts=np.asarray(time_stamps)
        ends=starts+stim_dur
    
        df,df_mean,count=create_df_from_timestamps(data=data,masks=masks,starts=starts,ends=ends,baseline_period=baseline_period,response_period=response_period, before=before,after=after,mouse_id=mouse_id,min_t=stim_dur,
                                             other_signals=other_signals,total_roi_counter=roi_counter,thresh_signal=False, use_dff=use_dff,sample_freq=30., )
        
        dfs.append(df)
        df_means.append(df_mean)
        
        
        roi_counter+=count
        
    
    returnd_df=pd.concat(dfs,axis=0,ignore_index=True)
    returnd_df_mean=pd.concat(df_means,axis=0,ignore_index=True)
    #clean up dataframes by re-indexing by the roi_ids
    returnd_df=returnd_df.sort_values(['roi id', 'epoch number'])
#    returnd_df.reset_index(drop=True)

    returnd_df_mean=returnd_df_mean.sort_values(['roi id'])
#    returnd_df_mean.reset_index(drop=True)
    
    return returnd_df, returnd_df_mean
        
    