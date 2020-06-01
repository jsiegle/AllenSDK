import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit
import logging

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, get_fr

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
            self._stim_table_nm1 = extract_movie_stim_table('natural_movie_one')
        return self._stim_table_nm1

    @property
    def stim_table_nm3(self):
        if self._stim_table_nm3 is None:
            self._stim_table_nm3 = extract_movie_stim_table('natural_movie_three')
        return self._stim_table_nm3


    def extract_movie_stim_table(self, movie_name):

        stim_table = self.ecephys_session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 
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
                ('sig_fraction_shuffle_nm3', np.float64)
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

            metrics_df['peak_frame_nm1'] = [self._get_peak_frame_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['peak_frame_nm3'] = [self._get_peak_frame_nm(unit, 'natural_movie_three') for unit in unit_ids]

            metrics_df['sig_fraction_shuffle_nm1'] = [self.responsiveness_vs_shuffle_nm(unit, 'natural_movie_one') for unit in unit_ids]
            metrics_df['sig_fraction_shuffle_nm3'] = [self.responsiveness_vs_shuffle_nm(unit, 'natural_movie_three') for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics

    @classmethod
    def known_stimulus_keys(cls):
        return ['natural_movies', 'natural_movie_one', 'natural_movie_three']

    def _get_stim_table_stats(self):
        pass
