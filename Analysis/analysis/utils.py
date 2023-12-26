import numpy as np
import pandas as pd
import xarray as xr
import h5py

STIMULUS_CONFIG = {
    'baseline': 'config.json',
    'short': 'config_short.json',
    'long': 'config_long.json',
    'const': 'config_const.json'
}

def stimulus_type_from_trial_name(trial_name):
    stim_type = next(s for s in trial_name.split('_') if s in STIMULUS_CONFIG)
    return stim_type, STIMULUS_CONFIG[stim_type]


def load_spikes_to_df(spike_file, network_name):
    with h5py.File(spike_file) as f:
        spikes_df = pd.DataFrame({
            'node_ids': f['spikes'][network_name]['node_ids'],
            'timestamps': f['spikes'][network_name]['timestamps']
        })
        spikes_df.sort_values(by='timestamps', inplace=True, ignore_index=True)
    return spikes_df


def load_ecp_to_xarray(ecp_file):
    with h5py.File(ecp_file, 'r') as f:
        ecp = xr.DataArray(
            f['ecp']['data'][()].T,
            coords = dict(
                channel_id = f['ecp']['channel_id'][()],
                time = np.arange(*f['ecp']['time']) # ms
            ),
            attrs = dict(
                fs = 1000 / f['ecp']['time'][2] # Hz
            )
        )
    return ecp

