#Author: Rylan Larsen

import numpy as np
import os
import h5py
import pandas as pd
    
from AxonImaging import signal_processing as sp

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
    stamps=sp.threshold_greater(np.array(stdev_filtered),threshold)

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
            median=np.nanmedian(sp.butter_lowpass_filter(data[thresh_signal], cutoff=1., analog=True))
            threshold_per=median+(thresh*median)
            thresh=threshold_per
            
        if exclusion_sig=='null':
            runs=sp.threshold_period(signal=data[thresh_signal], threshold=thresh,
                                  min_low=min_l, sample_freq=30., min_time=min_t)
            
        else:
            print (exclusion_logic+'  epochs where the '+ str(exclusion_sig) + ' is greater than '+ str(exclusion_thresh))
            
            runs=sp.threshold_period(signal=data[thresh_signal], threshold=thresh,min_time_between=min_time_between,
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
                
                runnings=sp.get_event_trig_avg_samples(data[thresh_signal],event_onset_times=starts[xx],
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
                
                axon_responses=sp.get_event_trig_avg_samples(data['axon_traces'][roi],event_onset_times=starts[xx],
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
                    sig=sp.get_event_trig_avg_samples(data[extras],event_onset_times=starts[xx],
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
            
            pvalue=sp.significant_response(before_after_mean, base_period=(baseline_period[0],baseline_period[1]), stim_period=(response_period[0],response_period_end), sample_freq=30.)    
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
                    runnings=sp.get_event_trig_avg_samples(data[thresh_signal],event_onset_times=ends[xx], event_end_times=ends[xx]+1,
                                                    sample_freq=sample_freq,time_before=before, time_after=after, verbose=False)
                else:     
                    #get the signal trace that was thresholded
                    runnings=sp.get_event_trig_avg_samples(data[thresh_signal],event_onset_times=starts[xx],
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
            axon_responses=sp.get_event_trig_avg_samples(data['axon_traces'][roi],event_onset_times=starts[xx],
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
                sig=sp.get_event_trig_avg_samples(data[extras],event_onset_times=starts[xx],
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
        
        pvalue=sp.significant_response(before_after_mean, base_period=(baseline_period[0],baseline_period[1]), stim_period=(response_period[0],response_period_end), sample_freq=30.)    
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
            median=np.nanmedian(sp.butter_lowpass_filter(data[thresh_signal], cutoff=1., analog=True))
            threshold_per=median+(thresh*median)
            thresh=threshold_per
            
        if exclusion_sig=='null':
            runs=sp.threshold_period(signal=data[thresh_signal], threshold=thresh,
                                  min_low=min_l, sample_freq=30., min_time=min_t)
            
        else:
            print (exclusion_logic+'  epochs where the '+ str(exclusion_sig) + ' is greater than '+ str(exclusion_thresh))
            
            runs=sp.threshold_period(signal=data[thresh_signal], threshold=thresh,min_time_between=min_time_between,
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
        
    