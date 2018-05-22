# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:46:56 2018

@author: Rylan Larsen
"""

import numpy as np
import math


def threshold_greater(data, threshold=0):
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
            frame_dur=1./np.mean(np.diff(signal_ts))
            values.append(signal_trace[index:(index+int(round(total_dur*frame_dur)))])
        
        else:
            values.append(signal_trace[index])
    return values
        
def threshold_period(signal, threshold, min_low, sample_freq, irregular_sampled_ts=np.array([]),pre_thresh_time=False,low_threshold=False,min_time=0.01, trace_end=994.0, exclusion_signal=np.array([]), exclusion_thresh=0, exclusion_sf=0):
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
            exclusion_up_t=t_up_conv[ii] * exclusion_sf
            exclusion_d_t=t_down_conv[[ii]]* exclusion_sf
            if  np.mean(exclusion_signal[int(exclusion_up_t):int(exclusion_d_t)],axis=0)<exclusion_thresh:
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

    #measure duration of each period
    periods=[]
    for x in range(len(epoch_start)):
        
        duration=epoch_end[x]-epoch_start[x]
        if duration >= min_time:
            periods.append([epoch_start[x], epoch_end[x], duration])
    
    periods=np.array(periods)
    
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
        
    snr=[]
    
    for xx in range(len(traces)):
        power_spect=np.abs(np.fft.fft(traces[xx], axis=0))**2
        scaling=(sample_freq+1)*20
        signal=np.max(power_spect[int(signal_range[0]*scaling):int(signal_range[1]*scaling)])
        noise=np.mean(power_spect[int(noise_range[0]*scaling):int(noise_range[1]*scaling)])
        snr.append(math.log10(signal/noise))
        
    return np.asarray(snr)

 

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
        print 'Period Duration:', chunk_dur
        print 'Mean Duration of Each Frame:', mean_frame_dur
        print 'Number of frames per period:', chunk_frame_dur
            
    chunk_start_frame = int(np.floor(chunk_start/ mean_frame_dur))
    
    traces = []
    df_traces=[]
   
    
    if np.shape(event_onset_times):
        
        for x in range(len(event_onset_times)):
            onset_frame_ind= np.argmin(np.abs(frame_ts-event_onset_times[x]))
            chunk_start_frame_ind = onset_frame_ind + chunk_start_frame
            chunk_end_frame_ind = chunk_start_frame_ind + chunk_frame_dur
                    
            if verbose:
                print 'Period:',int(x),' Starting frame index:',chunk_start_frame_ind,'; Ending frame index', chunk_end_frame_ind
            
            if chunk_end_frame_ind <= trace.shape[0]:
                
                curr_trace = trace[chunk_start_frame_ind:chunk_end_frame_ind].astype(np.float32)
                traces.append(curr_trace)
                
                            
                df_curr_trace=stim_df_f(curr_trace, baseline_period=abs(chunk_start), frame_rate=30.)
                df_traces.append(df_curr_trace)
                
                
            else:
                print 'trace length',trace.shape[0],'is shorter than the referenced start time for the next stimulus', chunk_end_frame_ind
                continue
    
    else:
        print ('Only single stimulus period found, no averaging performed')
        onset_frame_ind= np.argmin(np.abs(frame_ts-event_onset_times))
        chunk_start_frame_ind = onset_frame_ind + chunk_start_frame
        chunk_end_frame_ind = chunk_start_frame_ind + chunk_frame_dur
        
        curr_trace=trace[chunk_start_frame_ind:chunk_end_frame_ind].astype(np.float32)
        traces = curr_trace
        df_traces=stim_df_f(curr_trace, baseline_period=chunk_start, frame_rate=30.)
        
    if dff==True:
        return np.asarray(df_traces)
    elif dff==False:
        return np.asarray(traces)     



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
            print 'Chunk:',int(n),' Starting frame index:',onset_frame_ind,'; Ending frame index', end_frame_ind
        
        if end_frame_ind <= trace.shape[0]:
            
            curr_trace = trace[onset_frame_ind:end_frame_ind].astype(np.float32)
            traces.append(curr_trace)
            n += 1
        else:
            print 'trace length',trace.shape[0],'is shorter than the referenced start time for the next stimulus', end_frame_ind
            continue
    return traces



def get_STA_traces_samples (trace, event_onset_times, chunk_start, total_dur, sample_freq, verbose=True):
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
        
        print '# samples per period:', chunk_dur
        print 'mean duration of each sample (seconds):', mean_sample_dur
    
    traces = []
    n = 0

    for x in range(len(event_onset_times)):
        
        #find the nearest analog sample to the event onset times
        onset_index=np.argmin(np.abs(sample_times-event_onset_times[x]))
        start_trace=onset_index + chunk_start
               
        end_trace=start_trace+chunk_dur
        
        if verbose:
            print 'Period:',int(n),' Starting frame index:',start_trace,'; Ending frame index', end_trace
        
        if end_trace <= trace.shape[0]:
            
            curr_trace = trace[start_trace:end_trace].astype(np.float32)
            traces.append(curr_trace)
            n += 1
            
        else:
            print 'trace length',trace.shape[0],'is shorter than the referenced start time for the next stimulus'
            continue
        
    return traces          

def get_event_trig_avg_samples (trace, event_onset_times, event_end_times, sample_freq, time_before=1, time_after=1, verbose=True):
    
    '''
    Get stimulus-triggered average(STA) trace for data where the units are not already in terms of seconds, but are in terms of continous samples (analog). (variable periods, analog)
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
    
    traces = []
    n = 0

    for x in range(len(event_onset_times)):
        
        #find the nearest analog sample to the event onset times
        start_trace=np.argmin(np.abs(sample_times-(event_onset_times[x]-time_before)))
        end_trace=np.argmin(np.abs(sample_times-(event_end_times[x]+time_after)))
               
        if verbose:
            print 'Period:',int(n),' Starting frame index:',start_trace,'; Ending frame index', end_trace
        
        if end_trace <= trace.shape[0]:
            
            curr_trace = trace[start_trace:end_trace].astype(np.float32)
            traces.append(curr_trace)
            n += 1
        else:
            print 'trace length',trace.shape[0],'is shorter than the referenced start time for the next stimulus'
            continue

    return traces

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
                        any signal data 16.5 ms before and 16.5 ms after is averaged in to produce a single time point.

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

def norm_percentile (traces, percentile=10,return_percent=False):
    '''Calculates baseline normalized, subtracted trace (DF/F) by setting the baseline (F0) to a percentile of the trace.

	#param traces: list of values
	#parm percentile: percentile for normalization and subtraction

	#returns Baseline normalized, subtracted (DF/F) array for each trace in absolute or as a percentage
	'''
    
    dff = np.zeros_like(traces)
    
    for xx in range(len(traces)): 
        dff[xx] = (traces[xx] - np.absolute(np.percentile(traces[xx],percentile)) / np.absolute(np.percentile(traces[xx],percentile)))
        
    if return_percent==True:
    	return dff*100.
    elif return_percent==False:
    	return dff
    else:
    	print ('No valid units specified for how to return Df_F trace (abs or %). Please specify and re-run')
    	return


def stim_df_f (arr, baseline_period, frame_rate=30.):
    '''calculates delta f/f where f0 is the baseline period preceding a stimulus

    #param arr: array of raw fluorescence traces to calculate df/f from. This should already encompass the baseline period, stimulus period, and post-stimulus period.
    #param baseline_period: amount of time, in seconds, to use for the baseline period
    #param frame_rate: imaging frame rate, default=30. This is used in determining the number of frames to extract for the baseline period.'''
    arr=np.asarray(arr)
    baseline_frames=int(abs(baseline_period)*frame_rate)
        
    if arr.ndim==2:
        
        df_f=[stim_df_f(arr=row,baseline_period=baseline_period, frame_rate=frame_rate) for row in arr]

    elif arr.ndim==3:

        df_f=[[stim_df_f(arr=n,baseline_period=baseline_period, frame_rate=frame_rate) for n in row] for row in arr]

    elif arr.ndim==1:
        f_o=np.mean(arr[0:baseline_frames-1],dtype=np.float32)
        delta_f=np.subtract(arr,f_o,dtype=np.float32)
        df_f=np.true_divide(delta_f,f_o)
    
    return df_f

def batch_stimulus_traces(signals, onsets, pre_gap,stim_dur,post_gap,df_f=False):
    ''' Returns a stimulus/event triggered response across many signals, all aligned to the same onset. Example, at the onset
    of a sound, align other time-series to the onset and include an arbituary amount of time before and after.
    
    :param stim_dict: a dictionary of t-series tuples in the format: 'key':(signal_trace,time_stamps (digital) OR sampling frequency)
                     EXAMPLE: 'microphone': (microphone, samplefreq) OR 'green_2P_traces':(green_traces,frame_times)
    :param onsets: timestamps that each of the t-series should be aligned relative to (seconds)
    :param pre_gap: relative time BEFORE each time step to include in the onset-aligned trace
    :param chunkStart: duration of the onset(stimulus). 
    :param post_gap: relative time AFTER each time step to include in the onset-aligned trace
    :param dff: whether to return values in terms of change from baseline normalized values (df/f).
                ONLY applies to digitally-sampled signals (w/ time stamps) who are multidimensional arrays.
    
    :return: averaged trace of all chunks
    '''
    total_time=abs(pre_gap)+post_gap+stim_dur
    
    stimulus_traces={}
    for keys in signals:
        #check to see if the second element is a int or float. If so, a sampling frequency has been provided and the signal can
        #be assumed to be continous (no-timestamps, analog)
        if isinstance(signals[keys][1], float) or isinstance(signals[keys][1], int) :
            trace=get_STA_traces_samples(signals[keys][0],event_onset_times=onsets,sample_freq=signals[keys][1],
                                            chunk_start=pre_gap,total_dur=total_time,verbose=False)
            time=np.linspace(start=0,stop=np.mean(trace,axis=0).size/signals[keys][1],num=np.mean(trace,axis=0).size)
            stimulus_traces[keys] = (time,trace)
            
        #if the second tuple element is a list or an array, then assume timestamps have been provided and that it is a digital
        #signal
        elif isinstance(signals[keys][1], list) or isinstance(signals[keys][1], np.ndarray):
            
            #check to see if array multi-dimensional, if so, run through loop
            if np.asarray(signals[keys][0]).ndim>1:
                stim_traces=[]
                sample_dur=1/np.mean(np.diff(signals[keys][1]))
                for xx in range(len(signals[keys][0])-1):
                    stim_traces.append(get_STA_traces_stamps(signals[keys][0][xx],frame_ts=signals[keys][1],
                                                                event_onset_times=onsets, 
                                                                chunk_start=pre_gap, 
                                                                chunk_dur=total_time,dff=df_f, verbose=False))
        
                time=np.linspace(start=0,stop=np.nanmean(np.nanmean(stim_traces,axis=0),axis=0).size/sample_dur,
                                 num=np.nanmean(np.nanmean(stim_traces,axis=0),axis=0).size)
            
                stimulus_traces[keys] = (time,stim_traces)
                
            else:
            
                trace=get_STA_traces_stamps(signals[keys][0],frame_ts=signals[keys][1], event_onset_times=onsets,
                                           chunk_start=pre_gap, chunk_dur=total_time,dff=False, verbose=False)
                
                sample_dur=1/np.mean(np.diff(signals[keys][1]))
                time=np.linspace(start=0,stop=np.nanmean(trace,axis=0).size/sample_dur,num=np.nanmean(trace,axis=0).size)
                stimulus_traces[keys] = (time,trace)
            
        else:
            print ('invalid data types provided. Please input a dictionary containing tuples')
            break
            return
    #return a dictionary with keys as the signal names, and the values are tuples of the format (corresponding time_array, stim_trace)
    return stimulus_traces