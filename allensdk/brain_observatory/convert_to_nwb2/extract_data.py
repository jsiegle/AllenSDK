import os
import numpy as np
import pandas as pd
import json, io
from scipy.ndimage.measurements import center_of_mass

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import EpochSeparationException


def extract_data_from_nwb1(experiment_id, manifest_path, temp_directory):
    
    """ Retrieves data from an NWB v1.0 file (using the AllenSDK) and writes
    it to a temporary directory.
    
    Parameters:
    -----------
    experiment_id : int
    manifest_path : filepath (str)
    temp_directory : directory path (str)
    
    Returns:
    --------
    filepaths : dict
        Dictionary of data file locations
    
    """
    
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    
    data_set = boc.get_ophys_experiment_data(experiment_id)
    events = boc.get_ophys_experiment_events(experiment_id)
    
    filepaths = {
        'stim_table' : os.path.join(temp_directory, 'stim_table.csv'),
        'event_times' : os.path.join(temp_directory, 'event_times.npy'),
        'event_cell_ids' : os.path.join(temp_directory, 'event_cell_ids.npy'),
        'event_amplitudes' : os.path.join(temp_directory, 'event_amplitudes.npy'),
        'running_speed' : os.path.join(temp_directory, 'running_speed.h5'),
        'metadata' : os.path.join(temp_directory, 'metadata.json'),
        'cell_info' : os.path.join(temp_directory, 'cell_info.csv'),
        'roi_masks' : os.path.join(temp_directory, 'roi_masks.npy'),
        'max_projection' : os.path.join(temp_directory, 'max_projection.npy'),
        'raw_traces' : os.path.join(temp_directory, 'raw_traces.npy'),
        'demixed_traces' : os.path.join(temp_directory, 'demixed_traces.npy'),
        'corrected_traces' : os.path.join(temp_directory, 'corrected_traces.npy'),
        'neuropil_traces' : os.path.join(temp_directory, 'neuropil_traces.npy'),
        'dff_traces' : os.path.join(temp_directory, 'dff_traces.npy'),
        'trace_times' : os.path.join(temp_directory, 'trace_times.npy')     
        }
    
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)
    
    metadata = data_set.get_metadata()
    metadata['session_start_time'] = metadata['session_start_time'].strftime("%Y-%m-%d %H:%M:%S")
    
    write_metadata(metadata, filepaths['metadata'])
    
    roi_mask_array = data_set.get_roi_mask_array()
    
    write_roi_masks(roi_mask_array, filepaths['roi_masks'])
    
    max_projection = data_set.get_max_projection()
    
    write_max_projection(max_projection, filepaths['max_projection'])
    
    cell_ids = data_set.get_cell_specimen_ids()
    
    time, raw_traces = data_set.get_fluorescence_traces(cell_specimen_ids=cell_ids)
    _, demixed_traces = data_set.get_demixed_traces(cell_specimen_ids=cell_ids)
    _, neuropil_traces = data_set.get_neuropil_traces(cell_specimen_ids=cell_ids)
    _, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=cell_ids)
    _, dff_traces = data_set.get_dff_traces(cell_specimen_ids=cell_ids)
    
    write_traces(raw_traces, filepaths['raw_traces'])
    write_traces(demixed_traces, filepaths['demixed_traces'])
    write_traces(neuropil_traces, filepaths['neuropil_traces'])
    write_traces(corrected_traces, filepaths['corrected_traces'])
    write_traces(dff_traces, filepaths['dff_traces'])
    write_traces(time, filepaths['trace_times'])
    
    event_times = []
    event_cell_ids = []
    event_amplitudes = []
    
    center_dict = {}
    
    roi_mask_list = data_set.get_roi_mask(cell_specimen_ids=cell_ids)
    
    for cell_idx, cell_id in enumerate(cell_ids):
        
        event_inds = np.where(events[cell_idx,:] > 0)[0]
        event_times.append(time[event_inds])
        event_cell_ids.append([cell_ids[cell_idx]] * len(event_inds))
        event_amplitudes.append(events[cell_idx, event_inds])
        
        mask = roi_mask_list[cell_idx].get_mask_plane()
        
        x,y = center_of_mass(mask)
        
        if not cell_ids[cell_idx] in center_dict.keys():
            center_dict[cell_ids[cell_idx]] = {'x' : int(x), 'y' : int(y)}
    
    event_amplitudes = np.concatenate(event_amplitudes)
    event_times = np.concatenate(event_times)
    event_cell_ids = np.concatenate(event_cell_ids)
    
    order = np.argsort(event_times)
    
    event_amplitudes = event_amplitudes[order]
    event_times = event_times[order]
    event_cell_ids = event_cell_ids[order]
    
    write_event_data(event_times,
                     event_cell_ids,
                     event_amplitudes,
                     filepaths['event_times'],
                     filepaths['event_cell_ids'],
                     filepaths['event_amplitudes'])
    
    write_cell_info(center_dict, filepaths['cell_info'])
    
    dxcm, dxtime = data_set.get_running_speed()
    
    write_running_speed(dxcm, dxtime, filepaths['running_speed'])

    stimuli = data_set.list_stimuli()
    
    tables = []
    
    try:
        epoch_table = data_set.get_stimulus_epoch_table()
    except EpochSeparationException:
        error('Could not load epoch table')
    else:
    
        for stimulus_index, stim in enumerate(stimuli):
    
            stim_table = data_set.get_stimulus_table(stim)
            
            epochs = epoch_table[epoch_table.stimulus == stim]
            
            for index, row in epochs.iterrows():
                stim_table.loc[(stim_table.start >= row.start) & 
                           (stim_table.end <= row.end), 'stimulus_block'] = int(index)
            
            stim_table['Start'] = time[stim_table['start']]
            stim_table['End'] = time[stim_table['end']]
            
            if stim != 'spontaneous':
                stim_table['stimulus_name'] = stim
                stim_table['stimulus_index'] = stimulus_index
            
            stim_table = stim_table.rename(columns = {'frame' : 'Image',
                                        'temporal_frequency' : 'TF',
                                        'spatial_frequency' : 'SF',
                                        'orientation' : 'Ori'})
            
            tables.append(stim_table)
                
    
    write_stim_table(pd.concat(tables, sort=False), filepaths['stim_table'])

    return filepaths
    

def write_traces(array, output_file):
    
    """ Writes fluorescence trace array to a CSV file

    Parameters
    ----------
    array : np.ndarray

    output_file : filepath (str)

    """
    
    np.save(output_file, array)

def write_roi_masks(roi_mask_array, output_file):
    
    """ Writes ROI mask array to a CSV file

    Parameters
    ----------
    roi_mask_array : np.ndarray

    output_file : filepath (str)

    """
    
    np.save(output_file, roi_mask_array)
    
    
def write_max_projection(max_projection, output_file):
    
    """ Writes max projection array to a CSV file

    Parameters
    ----------
    max_projection : np.ndarray

    output_file : filepath (str)

    """
    
    np.save(output_file, max_projection)
    

def write_stim_table(stim_table, output_file):

    """ Writes stim table to a CSV file

    Parameters
    ----------
    stim_table : pd.DataFrame

    output_file : filepath (str)

    """
    
    stim_table.to_csv(output_file, index=False)
    


def write_event_data(event_times, 
                     event_cell_ids, 
                     event_amplitudes, 
                     event_times_file,
                     event_cell_ids_file,
                     event_amplitudes_file):

    """ Writes event data to numpy files

    Parameters
    ----------
    event_times : np.ndarray
    event_cell_ids : np.ndarray
    event_amplitudes : np.ndarray

    event_times_file : filepath (str)
    event_cell_ids_file : filepath (str)
    output_file : filepath (str)

    """
    
    np.save(event_times_file, event_times)
    np.save(event_cell_ids_file, event_cell_ids)
    np.save(event_amplitudes_file, event_amplitudes)
    

def write_running_speed(dxcm, dxtime, output_file):

    """ Writes running speed data to an H5 file
    
    WARNING: CURRENTLY USES PLACEHOLDER VALUES FOR 
    VSIG, VIN, AND DX_DEG

    Parameters
    ----------
    dxcm : np.ndarray
    dxtime : np.ndarray
  
    output_file : filepath (str)

    """
    
    vsig = np.empty(dxtime.shape)
    vin = np.empty(dxtime.shape)
    dx_deg = np.empty(dxtime.shape)
    
    raw_data = pd.DataFrame(
            {"vsig": vsig, "vin": vin, "frame_time": dxtime, "dx": dx_deg}
        )
    
    start_time = dxtime
    end_time = dxtime + 0.033
    velocity = dxcm
    net_rotation = np.empty(velocity.shape)
    
    velocities = pd.DataFrame(
            {'start_time': start_time, "end_time": end_time, 
             'velocity' : velocity, 'net_rotation' : net_rotation}
        )
    
    store = pd.HDFStore(output_file)
    store.put("running_speed", velocities)
    store.put("raw_data", raw_data)
    store.close()


def write_cell_info(center_dict, output_file):

    """ Writes cell metadata to a CSV file

    Parameters
    ----------
    center_dict : dictionary of ROI centers
    output_file : filepath (str)

    """
    
    x_pos = []
    y_pos = []
    
    cell_ids = list(center_dict.keys())

    for cell_id in cell_ids:
        x_pos.append(center_dict[cell_id]['x'])
        y_pos.append(center_dict[cell_id]['y'])
    
    df = pd.DataFrame(data = {'cell_id' : cell_ids, 'x_pos' : x_pos, 'y_pos' : y_pos})
    df.to_csv(output_file, index=False)


def write_metadata(metadata_dictionary, output_file):

    """ Writes metadata dictionary to a JSON file.

    Parameters
    ----------
    metadata_dictionary : dict
    output_file : filepath (str)

    """
    
    with io.open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(metadata_dictionary, ensure_ascii=False, sort_keys=True, indent=4))

