from marshmallow import RAISE

from argschema import ArgSchema
from argschema.fields import (
    LogLevel,
    Dict,
    String,
    Int,
    DateTime,
    Nested,
    Boolean,
    Float,
)

from allensdk.brain_observatory.argschema_utilities import (
    check_read_access,
    check_write_access,
    RaisingSchema,
)


class Channel(RaisingSchema):
    id = Int(required=True)
    probe_id = Int(required=True)
    valid_data = Boolean(required=True)
    local_index = Int(required=True)
    probe_vertical_position = Int(required=False)
    probe_horizontal_position = Int(required=False)
    ecephys_structure_id = Int(required=False, allow_none=True)
    ecephys_structure_acronym = String(required=False, allow_none=True)


class Unit(RaisingSchema):
    id = Int(required=True)
    peak_channel_id = Int(required=True)
    local_index = Int(
        required=True,
        help="within-probe index of this unit.",
    )
    cluster_id = Int(
        required=True,
        help="within-probe identifier of this unit",
    )
    quality = String(required=True)
    firing_rate = Float(required=False)
    snr = Float(required=False, allow_none=True)
    isi_violations = Float(required=False)
    presence_ratio = Float(required=False)
    amplitude_cutoff = Float(required=False)
    isolation_distance = Float(required=False, allow_none=True)
    l_ratio = Float(required=False, allow_none=True)
    d_prime = Float(required=False, allow_none=True)
    nn_hit_rate = Float(required=False, allow_none=True)
    nn_miss_rate = Float(required=False, allow_none=True)
    max_drift = Float(required=False, allow_none=True)
    cumulative_drift = Float(required=False, allow_none=True)
    silhouette_score = Float(required=False, allow_none=True)
    waveform_duration = Float(required=False, allow_none=True)
    waveform_halfwidth = Float(required=False, allow_none=True)
    PT_ratio = Float(required=False, allow_none=False)
    repolarization_slope = Float(required=False, allow_none=True)
    recovery_slope = Float(required=False, allow_none=True)
    amplitude = Float(required=False, allow_none=True)
    spread = Float(required=False, allow_none=True)
    velocity_above = Float(required=False, allow_none=True)
    velocity_below = Float(required=False, allow_none=True)


class Lfp(RaisingSchema):
    input_data_path = String(required=True, validate=check_read_access)
    input_timestamps_path = String(required=True, validate=check_read_access)
    input_channels_path = String(required=True, validate=check_read_access)
    output_path = String(required=True)


class Probe(RaisingSchema):
    id = Int(required=True)
    name = String(required=True)
    spike_times_path = String(required=True, validate=check_read_access)
    spike_clusters_file = String(required=True, validate=check_read_access)
    channels = Nested(Channel, many=True, required=True)
    units = Nested(Unit, many=True, required=True)
    sampling_rate = Float(default=30000.0, help="sampling rate (Hz, master clock) at which raw data were acquired on this probe")
    lfp_sampling_rate = Float(default=2500.0, allow_none=True, help="sampling rate of LFP data on this probe")
    temporal_subsampling_factor = Float(default=2.0, allow_none=True, help="subsampling factor applied to lfp data for this probe (across time)")
    spike_amplitudes_path = String(validate=check_read_access, 
        help="path to npy file containing scale factor applied to the kilosort template used to extract each spike"
    )

class InvalidEpoch(RaisingSchema):
    id = Int(required=True)
    type = String(required=True)
    label = String(required=True)
    start_time = Float(required=True)
    end_time = Float(required=True)


class SessionMetadata(RaisingSchema):
      specimen_name = String(required=True)
      age_in_days = Float(required=True)
      full_genotype = String(required=True)
      strain = String(required=True)
      sex = String(required=True)
      stimulus_name = String(required=True)


class InputSchema(ArgSchema):
    class Meta:
        unknown = RAISE

    log_level = LogLevel(
        default="INFO", help="set the logging level of the module"
    )
    output_path = String(
        required=True,
        validate=check_write_access,
        help="write outputs to here",
    )
    session_id = Int(
        required=True, help="unique identifier for this ecephys session"
    )
    session_start_time = DateTime(
        required=True,
        help="the date and time (iso8601) at which the session started",
    )
    stimulus_table_path = String(
        required=True,
        validate=check_read_access,
        help="path to stimulus table file",
    )
    invalid_epochs = Nested(
        InvalidEpoch,
        many=True,
        required=False,
        help="epochs with invalid data"
    )
    probes = Nested(
        Probe,
        many=True,
        required=True,
        help="records of the individual probes used for this experiment",
    )
    running_speed_path = String(
        required=True,
        help="data collected about the running behavior of the experiment's subject",
    )
    session_sync_path = String(
        required=False,
        validate=check_read_access,
        help="Path to an h5 experiment session sync file (*.sync). This file relates events from different acquisition modalities to one another in time."
    )
    eye_tracking_rig_geometry = Dict(
        required=False,
        help="Mapping containing information about session rig geometry used for eye gaze mapping."
    )
    eye_dlc_ellipses_path = String(
        required=False,
        validate=check_read_access,
        help="h5 filepath containing raw ellipse fits produced by Deep Lab Cuts of subject eye, pupil, and corneal reflections during experiment"
    )
    eye_gaze_mapping_path = String(
        required=False,
        allow_none=True,
        help="h5 filepath containing eye gaze behavior of the experiment's subject"
    )
    pool_size = Int(
        default=3,
        help="number of child processes used to write probewise lfp files"
    )
    optotagging_table_path = String(
        required=False,
        validate=check_read_access,
        help="file at this path contains information about the optogenetic stimulation applied during this "
    )
    session_metadata = Nested(SessionMetadata, allow_none=True, required=False, help="miscellaneous information describing this session")


class ProbeOutputs(RaisingSchema):
    nwb_path = String(required=True)
    id = Int(required=True)


class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')
    probe_outputs = Nested(ProbeOutputs, required=True, many=True)
