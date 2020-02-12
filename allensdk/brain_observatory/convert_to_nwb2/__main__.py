import logging
import sys
from pathlib import Path, PurePath
import multiprocessing as mp
from functools import partial
import json
from datetime import datetime

import h5py
import pynwb
import requests
import pandas as pd
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO

from allensdk.config.manifest import Manifest

from ._schemas import InputSchema, OutputSchema
from allensdk.brain_observatory.nwb import (
    add_stimulus_presentations,
    add_stimulus_timestamps,
    add_invalid_times,
    setup_table_for_epochs,
    setup_table_for_invalid_times,
    read_eye_dlc_tracking_ellipses,
    read_eye_gaze_mappings,
    add_eye_tracking_ellipse_fit_data_to_nwbfile,
    add_eye_gaze_mapping_data_to_nwbfile,
    eye_tracking_data_is_valid
)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs, optional_lims_inputs
)
from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.ecephys.file_io.continuous_file import ContinuousFile
from allensdk.brain_observatory.ecephys.nwb import EcephysProbe, EcephysLabMetaData
from allensdk.brain_observatory.sync_dataset import Dataset
import allensdk.brain_observatory.sync_utilities as su

from .extract_data import extract_data_from_nwb1

STIM_TABLE_RENAMES_MAP = {"Start": "start_time", "End": "stop_time"}

def load_and_squeeze_npy(path):
    return np.squeeze(np.load(path, allow_pickle=False))


def fill_df(df, str_fill=""):
    df = df.copy()

    for colname in df.columns:
        if not pd.api.types.is_numeric_dtype(df[colname]):
            df[colname].fillna(str_fill)

        if np.all(pd.isna(df[colname]).values):
            df[colname] = [str_fill for ii in range(df.shape[0])]

        if pd.api.types.is_string_dtype(df[colname]):
            df[colname] = df[colname].astype(str)

    return df


def get_inputs_from_lims(host, ecephys_session_id, output_root, job_queue, strategy):
    """
     This is a development / testing utility for running this module from the Allen Institute for Brain Science's 
    Laboratory Information Management System (LIMS). It will only work if you are on our internal network.

    Parameters
    ----------
    ecephys_session_id : int
        Unique identifier for session of interest.
    output_root : str
        Output file will be written into this directory.
    job_queue : str
        Identifies the job queue from which to obtain configuration data
    strategy : str
        Identifies the LIMS strategy which will be used to write module inputs.

    Returns
    -------
    data : dict
        Response from LIMS. Should meet the schema defined in _schemas.py

    """

    uri = f"{host}/input_jsons?object_id={ecephys_session_id}&object_class=EcephysSession&strategy_class={strategy}&job_queue_name={job_queue}&output_directory={output_root}"
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and "error" in data:
        raise ValueError("bad request uri: {} ({})".format(uri, data["error"]))

    return data


def read_stimulus_table(path, column_renames_map=None):
    """ Loads from a CSV on disk the stimulus table for this session. Optionally renames columns to match NWB 
    epoch specifications.

    Parameters
    ----------
    path : str
        path to stimulus table csv
    column_renames_map : dict, optional
        if provided will be used to rename columns from keys -> values. Default renames 'Start' -> 'start_time' and 
        'End' -> 'stop_time'
    
    Returns
    -------
    pd.DataFrame : 
        stimulus table with applied renames

    """

    if column_renames_map is None:
        column_renames_map = STIM_TABLE_RENAMES_MAP

    ext = PurePath(path).suffix

    if ext == ".csv":
        stimulus_table = pd.read_csv(path)
    else:
        raise IOError(f"unrecognized stimulus table extension: {ext}")

    return stimulus_table.rename(columns=column_renames_map, index={})


def read_event_times_to_dictionary(
    event_times_path, event_cell_ids_path, local_to_global_unit_map=None
):
    """ Reads event times and assigned cell ids from npy files into a lookup table.

    Parameters
    ----------
    event_times_path : str
        npy file identifying, per event, the time at which that event occurred.
    event_cell_ids_path : str
        npy file identifying, per event, the cell id associated with that event
    local_to_global_unit_map : dict, optional
        Maps local cell ids to global cell ids

    Returns
    -------
    output_times : dict
        keys are cell identifiers, values are event time arrays

    """

    event_times = load_and_squeeze_npy(event_times_path)
    event_cell_ids = load_and_squeeze_npy(event_cell_ids_path)

    return group_1d_by_unit(event_times, event_cell_ids, local_to_global_unit_map)


def read_event_amplitudes_to_dictionary(
    event_amplitudes_path, event_cell_ids_path, 
    local_to_global_unit_map=None
):

    event_amplitudes = load_and_squeeze_npy(event_amplitudes_path)
    event_cell_ids = load_and_squeeze_npy(event_cell_ids_path)

    return group_1d_by_unit(event_amplitudes, event_cell_ids, local_to_global_unit_map)



def group_1d_by_unit(data, data_unit_map, local_to_global_unit_map=None):
    sort_order = np.argsort(data_unit_map, kind="stable")
    data_unit_map = data_unit_map[sort_order]
    data = data[sort_order]

    changes = np.concatenate(
        [
            np.array([0]),
            np.where(np.diff(data_unit_map))[0] + 1,
            np.array([data.size]),
        ]
    )

    output = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = data_unit_map[low]
        current = data[low:high]

        if local_to_global_unit_map is not None:
            if local_unit not in local_to_global_unit_map:
                #logging.warning(
                #    f"unable to find unit at local position {local_unit}"
                #)
                continue
            global_id = local_to_global_unit_map[local_unit]
            output[global_id] = current
        else:
            output[local_unit] = current

    return output


def add_metadata_to_nwbfile(nwbfile, metadata):
    # Need to update to handle ophys metadata
    nwbfile.add_lab_meta_data(
        EcephysLabMetaData(name="metadata", **metadata)
    )
    return nwbfile


def read_running_speed(path):
    """ Reads running speed data and timestamps into a RunningSpeed named tuple

    Parameters
    ----------
    path : str
        path to running speed store


    Returns
    -------
    tuple : 
        first item is dataframe of running speed data, second is dataframe of 
        raw values (vsig, vin, encoder rotation)

    """

    return (
        pd.read_hdf(path, key="running_speed"), 
        pd.read_hdf(path, key="raw_data")
    )


def add_ragged_data_to_dynamic_table(
    table, data, column_name, column_description=""
):
    """ Builds the index and data vectors required for writing ragged array data to a pynwb dynamic table

    Parameters
    ----------
    table : pynwb.core.DynamicTable
        table to which data will be added (as VectorData / VectorIndex)
    data : dict
        each key-value pair describes some grouping of data
    column_name : str
        used to set the name of this column
    column_description : str, optional
        used to set the description of this column

    Returns
    -------
    nwbfile : pynwb.NWBFile

    """

    idx, values = dict_to_indexed_array(data, table.id.data)
    del data

    table.add_column(
        name=column_name, description=column_description, data=values, index=idx
    )


DEFAULT_RUNNING_SPEED_UNITS = {
    "velocity": "cm/s",
    "vin": "V",
    "vsig": "V",
    "rotation": "radians"
}


def add_running_speed_to_nwbfile(nwbfile, running_speed, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    running_mod = pynwb.ProcessingModule("running", "running speed data")
    nwbfile.add_processing_module(running_mod)

    running_speed_timeseries = pynwb.base.TimeSeries(
        name="running_speed",
        timestamps=np.array([
            running_speed["start_time"].values, 
            running_speed["end_time"].values
        ]),
        data=running_speed["velocity"].values,
        unit=units["velocity"]
    )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"]
    )

    running_mod.add_data_interface(running_speed_timeseries)
    running_mod.add_data_interface(rotation_timeseries)

    return nwbfile


def add_raw_running_data_to_nwbfile(nwbfile, raw_running_data, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    raw_rotation_timeseries = pynwb.base.TimeSeries(
        name="raw_running_wheel_rotation",
        timestamps=np.array(raw_running_data["frame_time"]),
        data=raw_running_data["dx"].values,
        unit=units["rotation"]
    )

    vsig_ts = pynwb.base.TimeSeries(
        name="running_wheel_signal_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vsig"].values,
        unit=units["vsig"]
    )

    vin_ts = pynwb.base.TimeSeries(
        name="running_wheel_supply_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vin"].values,
        unit=units["vin"]
    )

    nwbfile.add_acquisition(raw_rotation_timeseries)
    nwbfile.add_acquisition(vsig_ts)
    nwbfile.add_acquisition(vin_ts)

    return nwbfile

def add_rois(nwbfile, metadata, filepaths):
    """ Add ROIs for all cells
    """

    device = pynwb.device.Device(metadata['device'])
    nwbfile.add_device(device)

    max_projection = np.load(filepaths['max_projection'])
    roi_masks = np.load(filepaths['roi_masks'])

    optical_channel = pynwb.ophys.OpticalChannel('optical_channel', 'description', 500.)

    imaging_plane = nwbfile.create_imaging_plane('imaging_plane', 
                                                 optical_channel, 
                                                 metadata['targeted_structure'],
                                                 device, 
                                                 float(metadata['excitation_lambda'][:3]), 
                                                 30., 
                                                 metadata['indicator'], 
                                                 metadata['targeted_structure'],
                                                 np.ones((512, 512, 3)), 
                                                 4.0, 
                                                 'manifold unit', 
                                                 'A frame to refer to')

    ophys_module = nwbfile.create_processing_module('ophys', 'contains optical physiology processed data')

    image_series = pynwb.ophys.TwoPhotonSeries(name='max_projection', 
                                               data=[max_projection], 
                                               imaging_plane=imaging_plane,
                                               dimension=[512, 512],
                                               rate=30.0)
    nwbfile.add_acquisition(image_series)

    img_seg = pynwb.ophys.ImageSegmentation()
    ophys_module.add_data_interface(img_seg)

    ps = img_seg.create_plane_segmentation('roi_segmentation',
                                           imaging_plane, 'roi_segmentation', image_series)

    for i in range(roi_masks.shape[0]):
        ps.add_roi(image_mask=roi_masks[i,:,:])

    roi_table_region = ps.create_roi_table_region('all ROIs', region=list(range(roi_masks.shape[0])))

    return nwbfile, ophys_module, roi_table_region


def add_traces(nwbfile, ophys_module, roi_table_region, filepaths):
    """ Add traces for all cells
    """

    fluorescence = pynwb.ophys.Fluorescence()
    ophys_module.add_data_interface(fluorescence)

    response_types = ['raw_traces', 'demixed_traces', 'neuropil_traces', 'corrected_traces', 'dff_traces']

    timestamps = np.load(filepaths['trace_times'])

    for response in response_types:

        data = np.load(filepaths[response])

        rrs = fluorescence.create_roi_response_series(response, 
                                            data.T, 
                                            rois=roi_table_region,
                                            unit='lumens',
                                            timestamps=timestamps)

    return nwbfile

def add_events(nwbfile, filepaths):
    """ Adds event data for all cells
    """

    event_times = {}
    event_amplitudes = {}

    

    event_times.update(read_event_times_to_dictionary(
        filepaths['event_times'], filepaths['event_cell_ids']
    ))

    event_amplitudes.update(read_event_amplitudes_to_dictionary(
        filepaths["event_amplitudes"], filepaths["event_cell_ids"]
    ))

    cells_table = pd.read_csv(filepaths['cell_info'], index_col=0)
    cells_table.index.names = ['id']

    nwbfile.units = pynwb.misc.Units.from_dataframe(cells_table, name='units')

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=event_times,
        column_name="event_times",
        column_description="times (s) of detected L0 events",
    )

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=event_amplitudes,
        column_name="event_amplitudes",
        column_description="amplitudes of detected L0 events"
    )

    return nwbfile


def write_ophys_nwb(
    filepaths,
    experiment_id,
    output_path
):

    with open(filepaths['metadata'], 'r') as f:
        metadata = json.load(f)

    nwbfile = pynwb.NWBFile(
        session_description='OphysSession',
        identifier='{}'.format(experiment_id),
        session_start_time=datetime.strptime(metadata['session_start_time'], "%Y-%m-%d %H:%M:%S")
    )

    #print('Adding metadata...')
    #nwbfile = add_metadata_to_nwbfile(nwbfile, metadata)

    print('Adding rois...')
    nwbfile, ophys_module, roi_table_region = add_rois(nwbfile, metadata, filepaths)

    print('Adding traces...')
    nwbfile = add_traces(nwbfile, ophys_module, roi_table_region, filepaths)

    print('Adding events...')
    nwbfile = add_events(nwbfile, filepaths)    

    print('Adding stim table...')
    stimulus_table = read_stimulus_table(filepaths['stim_table'])
    nwbfile = add_stimulus_timestamps(nwbfile, stimulus_table['start_time'].values) # TODO: patch until full timestamps are output by stim table module
    nwbfile = add_stimulus_presentations(nwbfile, stimulus_table)

    print('Adding running speed...')
    running_speed, raw_running_data = read_running_speed(filepaths['running_speed'])
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    print('Writing NWB')
    Manifest.safe_make_parent_dirs(output_path)
    io = pynwb.NWBHDF5IO(output_path, mode='w')
    logging.info(f"writing session nwb file to {output_path}")
    io.write(nwbfile)
    io.close()

    return {
        'nwb_path': output_path
    }


def main():
    logging.basicConfig(
        format="%(asctime)s - %(process)s - %(levelname)s - %(message)s"
    )

    parser = optional_lims_inputs(sys.argv, InputSchema, OutputSchema, get_inputs_from_lims)

    filepaths = extract_data_from_nwb1(parser.args['experiment_id'],
                                       parser.args['manifest_path'],
                                       parser.args['temp_directory'])

    output = write_ophys_nwb(filepaths, parser.args['experiment_id'], parser.args['output_path'])

    write_or_print_outputs(output, parser)


if __name__ == "__main__":
    main()