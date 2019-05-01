import warnings
from typing import Dict, Union, List

import pandas as pd
import numpy as np
import pynwb
import xarray as xr

from .ecephys_api import EcephysApi
from allensdk.brain_observatory.nwb.nwb_api import NwbApi


class EcephysNwbApi(NwbApi, EcephysApi):

    def get_probes(self) -> pd.DataFrame:
        probes: Union[List, pd.DataFrame] = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({'id': int(k), 'description': v.description, 'location': v.location})
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        return probes

    def get_channels(self) -> pd.DataFrame:
        channels = self.nwbfile.electrodes.to_dataframe()
        channels.drop(columns='group', inplace=True)
        return channels

    def get_mean_waveforms(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['waveform_mean'].to_dict()

    def get_spike_times(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['spike_times'].to_dict()

    def get_units(self) -> pd.DataFrame:
        units_table = self._get_full_units_table()
        units_table.drop(columns=['spike_times', 'waveform_mean'], inplace=True)

        return units_table

    def get_lfp(self, probe_id: int, close: bool=True) -> xr.DataArray:
        lfp = self.nwbfile.get_acquisition(f'probe_{probe_id}_lfp')
        series = lfp.get_electrical_series(f'probe_{probe_id}_lfp_data')

        electrodes = pd.DataFrame(
            data=[ecr for ecr in series.electrodes], 
            columns=['id'] + list(series.electrodes.table.colnames)
        )

        data = series.data[:]
        timestamps = series.timestamps[:]

        if close:
            series.data.file.close()
            series.timestamps.file.close()

        return xr.DataArray(
            data=data,
            dims=['time', 'channel'],
            coords=[timestamps, electrodes['id'].values]
        )

    def get_ecephys_session_id(self) -> int:
        return int(self.nwbfile.identifier)

    def _get_full_units_table(self) -> pd.DataFrame:
        table = self.nwbfile.units.to_dataframe()
        table.index = table.index.astype(int)
        return table
