#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:49:12 2021

@author: danielm
"""

import os, warnings
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from sync3 import Dataset
import chisq_categorical as chi

SIG_THRESH = 0.01

def whiteboard(savepath):
    
    stim_table = create_stim_tables(savepath)['drifting_gratings']
    print(stim_table)
    
    dff_traces = load_dff_traces(savepath)
    print(np.shape(dff_traces))
    
    msr = load_mean_sweep_responses(savepath)
    print(np.shape(msr))    

    p_vals = chi_square_all_conditions(stim_table,msr,savepath)
    print(p_vals)
    
    sig_cells = p_vals < SIG_THRESH
    print('sig_cells: '+str(sig_cells.sum())+' of '+str(len(sig_cells)))
    
    condition_responses, blank_sweep_responses = compute_mean_condition_responses(stim_table,msr)
    condition_NLL, blank_NLL = compute_blank_subtracted_NLL(savepath)
    
    plot_single_cell_tuning(condition_NLL,blank_NLL,p_vals,savepath)
    
    SbC_pvals = test_SbC(savepath)
    frac_SbC = (SbC_pvals < 0.05).mean()
    print('Fraction suppressed by contrast: '+str(round(frac_SbC,3)))
    
    reliability = compute_reliability(savepath)
    print(reliability)
    
def plot_single_cell_tuning(condition_responses,blank_responses,p_vals,savepath):
    
    num_neurons = condition_responses.shape[0]
    directions, TFs = grating_params()
    
    for nc in range(num_neurons):
        plt.figure(figsize=(10,10))
        
        ax = plt.subplot(111)
        ax.imshow(condition_responses[nc],cmap='RdBu_r',vmin=-5,vmax=5,interpolation='none')
        
        ax.set_xticks(np.arange(len(TFs)))
        ax.set_xticklabels([str(int(x)) for x in TFs])
        ax.set_yticks(np.arange(len(directions)))
        ax.set_yticklabels([str(int(x)) for x in directions])
        
        ax.set_xlabel('TF')
        ax.set_ylabel('Direction')
        
        ax.set_title('p_val: '+str(p_vals[nc]))
        
        plt.savefig(savepath+'tuning_'+str(nc)+'.png',dpi=300)
        plt.close()
    
def chi_square_all_conditions(sweep_table,mean_sweep_events,savepath):
    
    if os.path.isfile(savepath+'chisq_all.npy'):
        p_vals = np.load(savepath+'chisq_all.npy')
    else:
        p_vals = chi.chisq_from_stim_table(sweep_table,
                                           ['Ori','TF'],
                                           mean_sweep_events,
                                           verbose=True)
        
        np.save(savepath+'chisq_all.npy',p_vals)
    
    return p_vals
    
def compute_reliability(savepath):
    
    dff_traces = load_dff_traces(savepath)
    msr = load_mean_sweep_responses(savepath)
    
    stim_table = create_stim_tables(savepath)['drifting_gratings']
    start_frame, end_frame = get_spontaneous_stim_table(savepath)
    
    condition_responses, blank_sweep_responses = compute_mean_condition_responses(stim_table,msr)
    peak_dir, peak_TF = get_peak_conditions(condition_responses)
    
    spont_traces = dff_traces[:,start_frame:end_frame]
    (num_neurons,num_frames) = np.shape(spont_traces)
    
    #build null distribution of trial responses from spontaneous activity
    num_shuffles = 5000
    null_sweeps = np.zeros((num_neurons,num_shuffles))
    for ns in range(num_shuffles):
        sweep_start = np.random.choice(num_frames-60)
        null_sweeps[:,ns] = np.mean(spont_traces[:,sweep_start:(sweep_start+60)],axis=1)
        
    #compare each trial for peak DG condition to 95th percentile of null distribution
    null_sorted = np.sort(null_sweeps,axis=1)
    cutoff = int(0.95*num_shuffles)
    sig_cutoff = null_sorted[:,cutoff]
    
    #neuron's reliability is fraction of peak condition trials that are significant 
    sig_sweeps = np.zeros(np.shape(msr))
    for nc in range(num_neurons):
        sig_sweeps[:,nc] = msr[:,nc] > sig_cutoff[nc]
        
    condition_reliability, __ = compute_mean_condition_responses(stim_table,sig_sweeps)
    
    reliability = np.zeros((num_neurons,))
    for nc in range(num_neurons):
        reliability[nc] = condition_reliability[nc,peak_dir[nc],peak_TF[nc]]
        
    return reliability
        
def get_peak_conditions(condition_responses):
    
    (num_cells,num_directions,num_TFs) = np.shape(condition_responses)
    
    peak_direction = np.zeros((num_cells,),dtype=np.uint8)
    peak_TF = np.zeros((num_cells,),dtype=np.uint8)
    for nc in range(num_cells):
        cell_max = np.nanmax(condition_responses[nc])
        is_max = condition_responses[nc] == cell_max
        
        if is_max.sum()==1:
            (direction,TF) = np.argwhere(is_max)[0,:]
        else:
            print(str(is_max.sum())+' peaks')
            r = np.random.choice(is_max.sum())
            (direction,TF) = np.argwhere(is_max)
            print(np.shape(direction))
            direction = direction[r]
            TF = TF[r]
        peak_direction[nc] = direction
        peak_TF[nc] = TF
        
    return peak_direction, peak_TF
    
def compute_blank_subtracted_NLL(savepath,num_shuffles=200000):
    
    if os.path.isfile(savepath+'blank_subtracted_NLL.npy'):
        condition_NLL = np.load(savepath+'blank_subtracted_NLL.npy')
        blank_NLL = np.load(savepath+'blank_subtracted_blank_NLL.npy')
    else:
        sweep_table = create_stim_tables(savepath)['drifting_gratings']
        mean_sweep_events = load_mean_sweep_responses(savepath)
        
        (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
        
        condition_responses, blank_sweep_responses = compute_mean_condition_responses(sweep_table,mean_sweep_events)        
        condition_responses = np.swapaxes(condition_responses,0,2)
        condition_responses = np.swapaxes(condition_responses,0,1)
        
        directions, TFs = grating_params()
        
        # different conditions can have different number of trials...
        trials_per_condition, num_blanks = compute_trials_per_condition(sweep_table)
        unique_trial_counts = np.unique(trials_per_condition.flatten())
        
        trial_count_mat = np.tile(trials_per_condition,reps=(num_cells,1,1))
        trial_count_mat = np.swapaxes(trial_count_mat,0,2)
        trial_count_mat = np.swapaxes(trial_count_mat,0,1)
        
        blank_shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*num_blanks,))
        blank_shuffle_responses = mean_sweep_events[blank_shuffle_sweeps].reshape(num_shuffles,num_blanks,num_cells)
        blank_null_dist = blank_shuffle_responses.mean(axis=1)
        
        condition_NLL = np.zeros((len(directions),len(TFs),num_cells))
        for trial_count in unique_trial_counts:

            #create null distribution and compute condition NLL
            shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*trial_count,))
            shuffle_responses = mean_sweep_events[shuffle_sweeps].reshape(num_shuffles,trial_count,num_cells)
            
            null_diff_dist = shuffle_responses.mean(axis=1) - blank_null_dist
            actual_diffs = condition_responses.reshape(len(directions),len(TFs),1,num_cells) - blank_sweep_responses.reshape(1,1,1,num_cells)
            resp_above_null = null_diff_dist.reshape(1,1,num_shuffles,num_cells) < actual_diffs
            percentile = resp_above_null.mean(axis=2)
            NLL = percentile_to_NLL(percentile,num_shuffles)
        
            has_count = trial_count_mat == trial_count
            condition_NLL = np.where(has_count,NLL,condition_NLL)
            
        #repeat for blank sweeps
        blank_null_dist_2 = blank_null_dist[np.random.choice(num_shuffles,size=num_shuffles),:]
        blank_null_diff_dist = blank_null_dist_2 - blank_null_dist
        actual_diffs = 0.0
        resp_above_null = blank_null_diff_dist < actual_diffs
        percentile = resp_above_null.mean(axis=0)
        blank_NLL = percentile_to_NLL(percentile,num_shuffles)
        
        np.save(savepath+'blank_subtracted_NLL.npy',condition_NLL)
        np.save(savepath+'blank_subtracted_blank_NLL.npy',blank_NLL)
        
    condition_NLL = np.swapaxes(condition_NLL,0,2)
    condition_NLL = np.swapaxes(condition_NLL,1,2)
        
    return condition_NLL, blank_NLL

def test_SbC(savepath):
        
    TF_for_DG_in_BoB_idx = 0
    
    condition_responses, blank_responses = compute_blank_subtracted_NLL(savepath)
    peak_dir = np.argmax(condition_responses[:,:,TF_for_DG_in_BoB_idx],axis=1)
    
    SbC_NLL = []
    for nc,direction in enumerate(peak_dir):
        SbC_NLL.append(condition_responses[nc,direction,TF_for_DG_in_BoB_idx])
    
    SbC_pvals = NLL_to_percentile(np.array(SbC_NLL))
        
    return SbC_pvals  

def percentile_to_NLL(percentile,num_shuffles):
    
    percentile = np.where(percentile==0.0,1.0/num_shuffles,percentile)
    percentile = np.where(percentile==1.0,1.0-1.0/num_shuffles,percentile)
    NLL = np.where(percentile<0.5,
                   np.log10(percentile)-np.log10(0.5),
                   -np.log10(1.0-percentile)+np.log10(0.5))
    
    return NLL

def NLL_to_percentile(NLL):
    
    percentile = np.where(NLL<0.0,
                          10.0**(NLL+np.log10(0.5)),
                          1.0-10.0**(np.log10(0.5)-NLL))
    
    return percentile

def compute_trials_per_condition(sweep_table):
    
    directions, TFs = grating_params()
    trials_per_condition = np.zeros((len(directions),len(TFs)),dtype=np.int)
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_TF,TF in enumerate(TFs):
            is_TF = sweep_table['TF'] == TF
            is_condition = (is_direction & is_TF).values
            trials_per_condition[i_dir,i_TF] = is_condition.sum()

    num_blanks = np.isnan(sweep_table['Ori'].values).sum()

    return trials_per_condition, num_blanks
    
def compute_mean_condition_responses(sweep_table,mean_sweep_events):
    
    (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
    
    directions, TFs = grating_params()
    
    condition_responses = np.zeros((num_cells,len(directions),len(TFs)))
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_TF,TF in enumerate(TFs):
            is_TF = sweep_table['TF'] == TF
            is_condition = (is_direction & is_TF).values
        
            condition_responses[:,i_dir,i_TF] = np.mean(mean_sweep_events[is_condition],axis=0)
            #print(str(int(direction))+' '+str(int(TF))+': '+str(int(is_condition.sum())))
            
    is_blank = np.isnan(sweep_table['Ori'].values)
    blank_sweep_responses = np.mean(mean_sweep_events[is_blank],axis=0)
            
    return condition_responses, blank_sweep_responses

def grating_params():
    
    directions = np.arange(0,360,45)
    TFs = np.array([1.0,2.0,4.0,8.0,15.0])
    
    return directions, TFs

def load_dff_traces(savepath):
    for f in os.listdir(savepath):
        if f.endswith('_dff.h5'):
            dff_file = h5py.File(savepath+f,mode='r')
            dff_traces = np.array(dff_file['data'])
            dff_file.close()
            
            has_nans = np.isnan(dff_traces).sum(axis=1)>0
            dff_traces = dff_traces[~has_nans]
            
            return dff_traces
    return None
    
def load_mean_sweep_responses(savepath,stim_name='drifting_gratings'):
    
    if os.path.isfile(savepath+'mean_sweep_responses.npy'):
        msr = np.load(savepath+'mean_sweep_responses.npy')
    else:
    
        dff_traces = load_dff_traces(savepath)
        sweep_table = create_stim_tables(savepath)[stim_name]
        
        num_neurons = np.shape(dff_traces)[0]
        num_sweeps = len(sweep_table)
        
        msr = np.zeros((num_sweeps,num_neurons))
        for sweep in range(num_sweeps):
            start_frame = int(sweep_table['Start'][sweep])
            end_frame = int(sweep_table['End'][sweep])
            
            baseline = np.mean(dff_traces[:,(start_frame-30):start_frame],axis=1)
            response = np.mean(dff_traces[:,start_frame:end_frame],axis=1)
            
            msr[sweep] = response #- baseline
            
        np.save(savepath+'mean_sweep_responses.npy',msr)
    
    return msr

def get_spontaneous_stim_table(exptpath):
    
    data = load_stim(exptpath)
    twop_frames, _, _, _ = load_sync(exptpath)
    
    MAX_SWEEPS = 50000
    start_frames = np.zeros((MAX_SWEEPS,))
    end_frames = np.zeros((MAX_SWEEPS,))
    
    curr_sweep = 0
    for i_stim, stim_data in enumerate(data['stimuli']):
        timing_table, __, __ = get_sweep_frames(data,i_stim)
        stim_sweeps = len(timing_table)
        
        start_frames[curr_sweep:(curr_sweep+stim_sweeps)] = twop_frames[timing_table['start'],0]
        end_frames[curr_sweep:(curr_sweep+stim_sweeps)] = twop_frames[timing_table['end'],0]
        curr_sweep += stim_sweeps
        
    start_frames = start_frames[:curr_sweep]
    end_frames = end_frames[:curr_sweep]
    
    sort_idx = np.argsort(start_frames)
    start_frames = start_frames[sort_idx]
    end_frames = end_frames[sort_idx]
    
    intersweep_frames = start_frames[1:] - end_frames[:-1]
    spontaneous_blocks = np.argwhere(intersweep_frames>2000)[:,0]
    
    sp_start_frames = []
    sp_end_frames = []
    for sp_idx in spontaneous_blocks:
        sp_start_frames.append(end_frames[sp_idx])
        sp_end_frames.append(start_frames[sp_idx+1])
        
    #pick longest block
    longest_block = np.argmax(np.array(sp_end_frames) - np.array(sp_start_frames))    
    
    return int(sp_start_frames[longest_block]), int(sp_end_frames[longest_block])
    
def create_stim_tables(
    exptpath,
    stimulus_names=[
        'drifting_gratings'
    ],
    verbose=True,
    ):
    """Create a stim table from data located in folder exptpath.
    Tries to extract a stim_table for each stim type in stimulus_names and
    continues if KeyErrors are produced.
    Inputs:
        exptpath (str)
            -- Path to directory in which to look for experiment-related files.
        stimulus_names (list of strs)
            -- Types of stimuli to try extracting.
        verbose (bool, default True)
            -- Print information about progress.
    Returns:
        Dict of DataFrames with information about start and end times of each
        stimulus presented in a given experiment.
    """
    data = load_stim(exptpath)
    twop_frames, _, _, _ = load_sync(exptpath)


    stim_table_funcs = {
        'drifting_gratings': DG_table,
    }
    stim_table = {}
    for stim_name in stimulus_names:
        try:
            stim_table[stim_name] = stim_table_funcs[stim_name](
                data, twop_frames
            )
        except KeyError:
            if verbose:
                print(
                    (
                        'Could not locate stimulus type {} in {}'.format(
                            stim_name, exptpath
                        )
                    )
                )
            continue

    return stim_table
    
def DG_table(data, twop_frames, verbose=True):

    DG_idx = get_stimulus_index(data, 'grating')

    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, DG_idx
    )
    if verbose:
        print(
            'Found {} of {} expected sweeps'.format(
                actual_sweeps, expected_sweeps
            )
        )

    stim_table = pd.DataFrame(
        np.column_stack(
            (
                twop_frames[timing_table['start']],
                twop_frames[timing_table['end']],
            )
        ),
        columns=('Start', 'End'),
    )

    for attribute in ['TF', 'SF', 'Contrast']:
        stim_table[attribute] = get_attribute_by_sweep(
            data, DG_idx, attribute
        )[: len(stim_table)]
    stim_table['Ori'] = get_attribute_by_sweep(data, DG_idx, 'Ori')[
        : len(stim_table)
    ]

    return stim_table

def get_stimulus_index(data, stim_name):
    """Return the index of stimulus in data.
    Returns the position of the first occurrence of stim_name in data. Raises a
    KeyError if a stimulus with a name containing stim_name is not found.
    Inputs:
        data (dict-like)
            -- Object in which to search for a named stimulus.
        stim_name (str)
    Returns:
        Index of stimulus stim_name in data.
    """
    for i_stim, stim_data in enumerate(data['stimuli']):
        if stim_name in stim_data['stim_path']:
            return i_stim

    raise KeyError('Stimulus with stim_name={} not found!'.format(stim_name))


def get_display_sequence(data, stimulus_idx):

    display_sequence = np.array(
        data['stimuli'][stimulus_idx]['display_sequence']
    )
    pre_blank_sec = int(data['pre_blank_sec'])
    display_sequence += pre_blank_sec
    display_sequence *= int(data['fps'])  # in stimulus frames

    return display_sequence

def get_sweep_frames(data, stimulus_idx):

    sweep_frames = data['stimuli'][stimulus_idx]['sweep_frames']
    timing_table = pd.DataFrame(
        np.array(sweep_frames).astype(np.int), columns=('start', 'end')
    )
    timing_table['dif'] = timing_table['end'] - timing_table['start']

    display_sequence = get_display_sequence(data, stimulus_idx)

    timing_table.start += display_sequence[0, 0]
    for seg in range(len(display_sequence) - 1):
        for index, row in timing_table.iterrows():
            if row.start >= display_sequence[seg, 1]:
                timing_table.start[index] = (
                    timing_table.start[index]
                    - display_sequence[seg, 1]
                    + display_sequence[seg + 1, 0]
                )
    timing_table.end = timing_table.start + timing_table.dif
    expected_sweeps = len(timing_table)
    timing_table = timing_table[timing_table.end <= display_sequence[-1, 1]]
    timing_table = timing_table[timing_table.start <= display_sequence[-1, 1]]
    actual_sweeps = len(timing_table)

    return timing_table, actual_sweeps, expected_sweeps


def get_attribute_by_sweep(data, stimulus_idx, attribute):

    attribute_idx = get_attribute_idx(data, stimulus_idx, attribute)

    sweep_order = data['stimuli'][stimulus_idx]['sweep_order']
    sweep_table = data['stimuli'][stimulus_idx]['sweep_table']

    num_sweeps = len(sweep_order)

    attribute_by_sweep = np.zeros((num_sweeps,))
    attribute_by_sweep[:] = np.NaN

    unique_conditions = np.unique(sweep_order)
    for i_condition, condition in enumerate(unique_conditions):
        sweeps_with_condition = np.argwhere(sweep_order == condition)[:, 0]

        if condition > -1:  # blank sweep is -1
            attribute_by_sweep[sweeps_with_condition] = sweep_table[condition][
                attribute_idx
            ]

    return attribute_by_sweep


def get_attribute_idx(data, stimulus_idx, attribute):
    """Return the index of attribute in data for the given stimulus.
    Returns the position of the first occurrence of attribute. Raises a
    KeyError if not found.
    """
    attribute_names = data['stimuli'][stimulus_idx]['dimnames']
    for attribute_idx, attribute_str in enumerate(attribute_names):
        if attribute_str == attribute:
            return attribute_idx

    raise KeyError(
        'Attribute {} for stimulus_ids {} not found!'.format(
            attribute, stimulus_idx
        )
    )
        
def load_stim(exptpath, verbose=True):
    """Load stim.pkl file into a DataFrame.
    Inputs:
        exptpath (str)
            -- Directory in which to search for files with _stim.pkl suffix.
        verbose (bool)
            -- Print filename (if found).
    Returns:
        DataFrame with contents of stim pkl.
    """
    # Look for a file with the suffix '_stim.pkl'
    pklpath = None
    for f in os.listdir(exptpath):
        if f.endswith('_stim.pkl'):
            pklpath = os.path.join(exptpath, f)
            if verbose:
                print("Pkl file:", f)

    if pklpath is None:
        raise IOError(
            'No files with the suffix _stim.pkl were found in {}'.format(
                exptpath
            )
        )

    return pd.read_pickle(pklpath) 

def load_sync(exptpath, verbose=True):

    # verify that sync file exists in exptpath
    syncpath = None
    for f in os.listdir(exptpath):
        if f.endswith('_sync.h5'):
            syncpath = os.path.join(exptpath, f)
            if verbose:
                print("Sync file:", f)
    if syncpath is None:
        raise IOError(
            'No files with the suffix _sync.h5 were found in {}'.format(
                exptpath
            )
        )

    # load the sync data from .h5 and .pkl files
    d = Dataset(syncpath)
    # print d.line_labels

    # set the appropriate sample frequency
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']

    # get sync timing for each channel
    twop_vsync_fall = d.get_falling_edges('2p_vsync') / sample_freq
    stim_vsync_fall = (
        d.get_falling_edges('stim_vsync')[1:] / sample_freq
    )  # eliminating the DAQ pulse
    photodiode_rise = d.get_rising_edges('stim_photodiode') / sample_freq

    # make sure all of the sync data are available
    channels = {
        'twop_vsync_fall': twop_vsync_fall,
        'stim_vsync_fall': stim_vsync_fall,
        'photodiode_rise': photodiode_rise,
    }
    channel_test = []
    for chan in list(channels.keys()):
        # Check that signal is high at least once in each channel.
        channel_test.append(any(channels[chan]))
    if not all(channel_test):
        raise RuntimeError('Not all channels present. Sync test failed.')
    elif verbose:
        print("All channels present.")

    # test and correct for photodiode transition errors
    ptd_rise_diff = np.ediff1d(photodiode_rise)
    short = np.where(np.logical_and(ptd_rise_diff > 0.1, ptd_rise_diff < 0.3))[
        0
    ]
    medium = np.where(np.logical_and(ptd_rise_diff > 0.5, ptd_rise_diff < 1.5))[
        0
    ]
    ptd_start = 3
    for i in medium:
        if set(range(i - 2, i)) <= set(short):
            ptd_start = i + 1
    ptd_end = np.where(photodiode_rise > stim_vsync_fall.max())[0][0] - 1

    if ptd_start > 3 and verbose:
        print('ptd_start: ' + str(ptd_start))
        print("Photodiode events before stimulus start.  Deleted.")

    ptd_errors = []
    while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
        error_frames = (
            np.where(ptd_rise_diff[ptd_start:ptd_end] < 1.8)[0] + ptd_start
        )
        print("Photodiode error detected. Number of frames:", len(error_frames))
        photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
        ptd_errors.append(photodiode_rise[error_frames[-1]])
        ptd_end -= 1
        ptd_rise_diff = np.ediff1d(photodiode_rise)

    first_pulse = ptd_start
    stim_on_photodiode_idx = 60 + 120 * np.arange(0, ptd_end - ptd_start, 1)

    stim_on_photodiode = stim_vsync_fall[stim_on_photodiode_idx]
    photodiode_on = photodiode_rise[
        first_pulse + np.arange(0, ptd_end - ptd_start, 1)
    ]
    delay_rise = photodiode_on - stim_on_photodiode

    delay = np.mean(delay_rise[:-1])
    if verbose:
        print("monitor delay: ", delay)

    # adjust stimulus time to incorporate monitor delay
    stim_time = stim_vsync_fall + delay

    # convert stimulus frames into twop frames
    twop_frames = np.empty((len(stim_time), 1))
    for i in range(len(stim_time)):
        # crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        crossings = (
            np.searchsorted(twop_vsync_fall, stim_time[i], side='left') - 1
        )
        if crossings < (len(twop_vsync_fall) - 1):
            twop_frames[i] = crossings
        else:
            twop_frames[i : len(stim_time)] = np.NaN
            warnings.warn('Acquisition ends before stimulus.', RuntimeWarning)
            break

    return twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise

if __name__=='__main__':
    savepath = '/Users/danielm/Desktop/Gad2/PV_369496/'
    whiteboard(savepath)