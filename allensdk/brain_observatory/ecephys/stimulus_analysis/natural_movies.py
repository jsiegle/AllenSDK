import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit
import logging

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, overall_firing_rate

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


class NaturalMovies(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the natural movies stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        nm_analysis = NaturalMovies(session)

    or, alternatively, pass in the file path::
        nm_analysis = Flashes('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        nm_analysis = NaturalMovies(session, filter={'location': 'probeC', 'ecephys_structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = nm_analysis.metrics()

    TODO: Need to find a default trial_duration otherwise class will fail
    """

    def __init__(self, ecephys_session, is_ophys_session=False, trial_duration=None, **kwargs):
        super(NaturalMovies, self).__init__(ecephys_session, is_ophys_session=is_ophys_session, trial_duration=trial_duration, **kwargs)

        self._metrics = None
        self._stim_table_nm1 = None
        self._stim_table_nm3 = None
        self._nm1_response = None
        self._nm3_response = None
        self._spont_response = None
        self._lifetime_sparseness_nm1 = None
        self._lifetime_sparseness_nm3 = None

        if self._params is not None:
            self._params = self._params['natural_movies']
            self._stimulus_key = self._params['stimulus_key']
        #else:
        #    self._stimulus_key = 'natural_movies'

    @property
    def name(self):
        return 'Natural Movies'

    @property
    def null_condition(self):
        return -1

    @property
    def stim_table_nm1(self):
        if self._stim_table_nm1 is None:
            self._stim_table_nm1 = self.extract_movie_stim_table('natural_movie_one')
        return self._stim_table_nm1

    @property
    def stim_table_nm3(self):
        if self._stim_table_nm3 is None:
            self._stim_table_nm3 = self.extract_movie_stim_table('natural_movie_three')
        return self._stim_table_nm3


    @property
    def nm1_response(self):
        if self._nm1_response is None:
            self._nm1_response = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.arange(0,self.stim_table_nm1.duration.mean(),1/30),
                stimulus_presentation_ids=self.stim_table_nm1.index.values,
                unit_ids=self.unit_ids,
                use_amplitudes=self._use_amplitudes)
        return self._nm1_response

    @property
    def nm3_response(self):
        if self._nm3_response is None:
            self._nm3_response = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.arange(0,self.stim_table_nm3.duration.mean(),1/30),
                stimulus_presentation_ids=self.stim_table_nm3.index.values,
                unit_ids=self.unit_ids,
                use_amplitudes=self._use_amplitudes)
        return self._nm3_response

    @property
    def spont_response(self):
        if self._spont_response is None:
            self._spont_response = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.arange(0,self.stim_table_nm3.duration.mean()*5,1/30),
                stimulus_presentation_ids=self.stim_table_spontaneous.index.values[0],
                unit_ids=self.unit_ids,
                use_amplitudes=self._use_amplitudes)
        return self._spont_response

    def extract_movie_stim_table(self, movie_name):

        stim_table = self.ecephys_session.stimulus_presentations[self.ecephys_session.stimulus_presentations.stimulus_name == 
                                                movie_name]

        start_times = stim_table[stim_table.frame == 0].start_time.values
        stop_times = stim_table[stim_table.frame == np.max(stim_table.frame)].stop_time.values

        stim_table = stim_table[stim_table.frame == 0]
        stim_table['stop_time'] = stop_times
        stim_table['duration'] = stop_times - start_times

        return stim_table
    
    @property
    def METRICS_COLUMNS(self):
        return [('firing_rate_nm1', np.float64),
                ('firing_rate_nm3', np.float64),
                ('lifetime_sparseness_nm1', np.float64), 
                ('lifetime_sparseness_nm3', np.float64),
                ('reliability_nm1', np.float64),
                ('reliability_nm3', np.float64),
                ('peak_frame_nm1', np.float64),
                ('peak_frame_nm3', np.float64),
                ('sig_fraction_shuffle_nm1', np.float64),
                ('sig_fraction_shuffle_nm3', np.float64),
                ('sig_fraction_spont_nm1', np.float64),
                ('sig_fraction_spont_nm3', np.float64)
                ]

    @property
    def metrics(self):
        if self._metrics is None:
            logger.info('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids
            metrics_df = self.empty_metrics_table()
            
            metrics_df['firing_rate_nm1'] = [self._get_overall_firing_rate_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['firing_rate_nm3'] = [self._get_overall_firing_rate_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['lifetime_sparseness_nm1'] = [self._get_lifetime_sparseness_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['lifetime_sparseness_nm3'] = [self._get_lifetime_sparseness_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['reliability_nm1'] = [self._get_reliability_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['reliability_nm3'] = [self._get_reliability_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['peak_frame_nm1'] = [self._get_peak_frame_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['peak_frame_nm3'] = [self._get_peak_frame_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['sig_fraction_shuffle_nm1'] = [self.responsiveness_vs_shuffle_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['sig_fraction_shuffle_nm3'] = [self.responsiveness_vs_shuffle_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['sig_fraction_spont_nm1'] = [self.responsiveness_vs_spont_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['sig_fraction_spont_nm3'] = [self.responsiveness_vs_spont_nm(unit, 'natural_movie_three') for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics

    @classmethod
    def known_stimulus_keys(cls):
        return ['natural_movies', 'natural_movie_one', 'natural_movie_three']


    def _get_overall_firing_rate_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            stim_table = self.stim_table_nm3
        elif movie_name == 'natural_movie_one':
            stim_table = self.stim_table_nm1
        else:
            raise Exception('Movie name not recognized.')

        start_time_intervals = np.diff(stim_table['start_time'])
        trial_duration = stim_table['duration'].mean()

        interval_end_inds = np.concatenate((np.where(start_time_intervals > trial_duration * 2)[0],
                                            np.array([len(stim_table)-1])))
        interval_start_inds = np.concatenate((np.array([0]),
                                              np.where(start_time_intervals > trial_duration * 2)[0] + 1))

        block_starts = stim_table.iloc[interval_start_inds]['start_time'].values
        block_stops = stim_table.iloc[interval_end_inds]['stop_time'].values
 
        return overall_firing_rate(start_times=block_starts, stop_times=block_stops,
                                   spike_times=self.ecephys_session.spike_times[unit_id])


    def _get_peak_frame_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            response = self.nm3_response
        elif movie_name == 'natural_movie_one':
            response = self.nm1_response
        else:
            raise Exception('Movie name not recognized.')

        return np.argmax(response.sel(unit_id = unit_id).mean(dim='stimulus_presentation_id').data)


    def _get_reliability_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            response = self.nm3_response
        elif movie_name == 'natural_movie_one':
            response = self.nm1_response
        else:
            raise Exception('Movie name not recognized.')

        subset = response.sel(unit_id = unit_id)
        presentation_ids = subset.stimulus_presentation_id
        num_trials = len(presentation_ids)

        corr_matrix = np.zeros((num_trials,num_trials))

        for i in range(num_trials):
            for j in range(i+1,num_trials):
                r,p = st.pearsonr(subset.sel(stimulus_presentation_id=
                                          presentation_ids[i]), 
                                  subset.sel(stimulus_presentation_id=
                                          presentation_ids[j]))
                corr_matrix[i,j] = r

        inds = np.triu_indices(num_trials, k=1)
        reliability = np.nanmean(corr_matrix[inds[0],inds[1]])

        return reliability


    def _get_lifetime_sparseness_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            if self._lifetime_sparseness_nm3 is None:
                self._lifetime_sparseness_nm3 = lifetime_sparseness_nm(self.nm3_response)
            return self._lifetime_sparseness_nm3.sel(unit_id=[unit_id], drop=True).data[0]

        elif movie_name == 'natural_movie_one':
            if self._lifetime_sparseness_nm1 is None:
                self._lifetime_sparseness_nm1 = lifetime_sparseness_nm(self.nm1_response)
            return self._lifetime_sparseness_nm1.sel(unit_id=[unit_id], drop=True).data[0]

        else:
            raise Exception('Movie name not recognized.')

    def responsiveness_vs_shuffle_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            response = self.nm3_response
        elif movie_name == 'natural_movie_one':
            response = self.nm1_response
        else:
            raise Exception('Movie name not recognized.')

        sig_level_shuffle = np.quantile(response.sel(unit_id=unit_id).data.flatten(), 0.95)

        peak_response = response.sel(unit_id=unit_id).isel(time_relative_to_stimulus_onset=
                    self._get_peak_frame_nm(unit_id, movie_name)).data

        sig_fraction_shuffle = np.sum(peak_response > sig_level_shuffle) / len(peak_response)

        return sig_fraction_shuffle

    def responsiveness_vs_spont_nm(self, unit_id, movie_name):

        if movie_name == 'natural_movie_three':
            response = self.nm3_response
        elif movie_name == 'natural_movie_one':
            response = self.nm1_response
        else:
            raise Exception('Movie name not recognized.')

        sig_level_spont = np.quantile(self.spont_response.sel(unit_id = unit_id).data, 0.95)

        peak_response = response.sel(unit_id=unit_id).isel(time_relative_to_stimulus_onset=
                    self._get_peak_frame_nm(unit_id, movie_name)).data

        sig_fraction_spont = np.sum(peak_response > sig_level_spont) / len(peak_response)

        return sig_fraction_spont


def lifetime_sparseness_nm(response):

    num_frames = len(response.time_relative_to_stimulus_onset)

    lifetime_sparseness = (( 1 - (1/num_frames) *  
                   (
                       (np.power(response.mean(dim='stimulus_presentation_id').sum(dim='time_relative_to_stimulus_onset'), 2))                    
                   /
                       (np.power(response.mean(dim='stimulus_presentation_id'), 2).sum(dim='time_relative_to_stimulus_onset')) 
                    )) 
                   / 
                    (1 - (1 / num_frames)))

    return lifetime_sparseness

