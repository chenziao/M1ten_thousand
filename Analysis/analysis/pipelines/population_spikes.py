import numpy as np
import pandas as pd
import xarray as xr
import os
import warnings

from bmtool.util.util import load_nodes_from_paths
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from analysis import  utils, process
from build_input import get_populations


RESULT_PATH = "simulation_results"
OUTPUT_PATH = "analysis_results"
PN_SPK_PATH = os.path.join(OUTPUT_PATH, 'PN_spikes')
ITN_FR_PATH = os.path.join(OUTPUT_PATH, 'ITN_fr')
network_name = 'cortex'
PN_pop_names = ['CP', 'CS']
ITN_pop_names = ['FSI', 'LTS']

def set_variables(**kwargs):
    global_vars = globals()
    for key, value in kwargs.items():
        global_vars[key] = value
    # set up data path
    global PN_SPK_PATH, ITN_FR_PATH
    PN_SPK_PATH = os.path.join(OUTPUT_PATH, 'PN_spikes')
    ITN_FR_PATH = os.path.join(OUTPUT_PATH, 'ITN_fr')
    if not os.path.isdir(PN_SPK_PATH):
        os.mkdir(PN_SPK_PATH)
    if not os.path.isdir(ITN_FR_PATH):
        os.mkdir(ITN_FR_PATH)


def get_trials(filter=[], trial_list=None):
    """Get name of trials from simulation result directory
    filter: str or nested list of str. Get trials with matching substrings
        If is nested list, the inner lists apply union to the conditions
        and the outer list apply intersection to the conditions.
        E.g., [('a', 'b'), ('c', 'd'), 'e'], means (a OR b) AND (c OR d) AND e.
    trial_list: list of trial names. If not specified, obtain from result directory
    """
    if trial_list is None:
        trial_list = [f for f in os.listdir(RESULT_PATH) if f[0] != '.']
    if isinstance(filter, str):
        filter = [[filter]]
    elif len(filter) == 0:
        filter = [filter]
    ffilt = set(trial_list)
    for filt in filter:
        if isinstance(filt, str):
            filt = [filt]
        ffilt &= set(f for f in trial_list if any(s in f for s in filt))
    trial_list = [f for f in trial_list if f in ffilt]
    return trial_list


def get_file(trial_name):
    PN_spk_file = os.path.join(PN_SPK_PATH, trial_name + '.npz')
    ITN_fr_file = os.path.join(ITN_FR_PATH, trial_name + '.nc')
    return PN_spk_file, ITN_fr_file


def preprocess(trial_name, fs_ct=400., fs_fr=50., filt_sigma=20.0, overwrite=False):
    """Load spike data from simulation result, preprocess and save outputs
    trial_name: name of simulation trial
    fs_ct: spike count sampling frequency (Hz)
    fs_fr: firing rate sampling frequency (Hz)
    filt_sigma: Gaussian filer sigma (ms)
    overwrite: whether overwrite output data file if already exists
    """
    PN_spk_file, ITN_fr_file = get_file(trial_name)
    process_PN = overwrite or not os.path.isfile(PN_spk_file)
    process_ITN = overwrite or not os.path.isfile(ITN_fr_file)
    if not (process_PN or process_ITN):
        return

    # Load trial information
    trial_path = os.path.join(RESULT_PATH, trial_name)
    _, paths, stim_info, _ = utils.get_trial_info(trial_path)
    _, NODE_FILES, SPIKE_FILE = paths
    t_stop, _, stim_params = stim_info
    t_start = stim_params['t_start']

    # Load trial data
    pop_names = PN_pop_names * process_PN + ITN_pop_names * process_ITN
    node_df = load_nodes_from_paths(NODE_FILES)[network_name]
    pop_ids = get_populations(node_df, pop_names, only_id=True)
    spikes_df = utils.load_spikes_to_df(SPIKE_FILE, network_name)

    # Parameters
    time_ct_edge = np.linspace(0, 1000 * t_stop, int(t_stop * fs_ct), endpoint=False)
    time_ct = time_ct_edge + 1000 / fs_ct / 2
    i_start_ct = pd.Index(time_ct).get_indexer([1000 * t_start], method='bfill')[0]

    time_fr = np.linspace(0, 1000 * t_stop, int(t_stop * fs_fr), endpoint=False)
    time_fr = time_fr + 1000 / fs_fr / 2
    time_fr = time_fr[(time_fr >= time_ct[0]) & (time_fr <= time_ct[-1])]
    i_start_fr = pd.Index(time_fr).get_indexer([1000 * t_start], method='bfill')[0]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        # PN units spike counts and firing rate
        if process_PN:
            PN_node_ids = [pop_ids[p] for p in PN_pop_names]
            pop_slice = np.cumsum([0] + list(map(len, PN_node_ids))) 
            pop_slice = dict(zip(PN_pop_names, zip(pop_slice[:-1], pop_slice[1:])))
            PN_node_ids = np.concatenate(PN_node_ids)
            unit_fr, unit_ct = process.unit_spike_rate_to_xarray(spikes_df, time_ct_edge,
                PN_node_ids, frequeny=True, filt_sigma=filt_sigma, return_count=True)
            unit_fr = interp1d(time_ct, unit_fr, axis=1, assume_sorted=True)(time_fr)
            PN_spk = dict(
                unit_ct=unit_ct, time_ct=time_ct, fs_ct=fs_ct, i_start_ct=i_start_ct,
                unit_fr=unit_fr, time_fr=time_fr, fs_fr=fs_fr, i_start_fr=i_start_fr,
                nodes=PN_node_ids, gauss_filt_sigma=filt_sigma, t_start=t_start
            )
            np.savez_compressed(PN_spk_file, **PN_spk)

        # ITNs firing rate with gauss smoothing
        if process_ITN:
            ITN_rspk = process.group_spike_rate_to_xarray(spikes_df, time_ct_edge,
                {p: pop_ids[p] for p in ITN_pop_names}, group_dims='population')
            axis = ITN_rspk.spike_rate.dims.index('time')
            sigma = np.zeros(ITN_rspk.spike_rate.ndim)
            sigma[axis] = filt_sigma * fs_ct / 1000
            ITN_rspk.spike_rate[:] = gaussian_filter(ITN_rspk.spike_rate, sigma)
            ITN_rspk = ITN_rspk.spike_rate.interp({'time': time_fr}, assume_sorted=True)\
                .to_dataset(name='spike_rate')\
                .assign(population_number=ITN_rspk.population_number)\
                .assign_attrs(gauss_filt_sigma=filt_sigma,
                              fs=fs_fr, t_start=t_start, i_start=i_start_fr)
            ITN_rspk.to_netcdf(ITN_fr_file)


def PN_stp_weights(PN_spk_file, tau, data='fr'):
    """Compute STP weigths from preprocessed PN spike data
    tau: exponential filter time constant (second), scalar or array
    data: spike data used for exponential filtering.
        'fr' for smoothed firing rate, 'ct' for original spike count data.
    """
    tau = np.asarray(tau)
    with np.load(PN_spk_file) as f:
        unit_fr = f['unit_fr']
        i_start = f['i_start_fr']
        if data == 'ct':
            time_fr, time_ct = f['time_fr'], f['time_ct']
            unit_ct = f['fs_ct'] * f['unit_ct']
            fr_exp_filt = [process.exponential_spike_filter(unit_ct, tau=t * f['fs_ct'],
                min_rate=0, normalize=True, last_jump=False) for t in tau.ravel()]
            fr_exp_filt = interp1d(time_ct, fr_exp_filt, axis=-1, assume_sorted=True)(time_fr)
        else:
            fr_exp_filt = [process.exponential_spike_filter(unit_fr, tau=t * f['fs_fr'],
                min_rate=0, normalize=True, last_jump=False) for t in tau.ravel()]
        unit_fr = unit_fr[:, i_start:]
        fr_exp_filt = np.array(fr_exp_filt)[:, :, i_start:].reshape(tau.shape + unit_fr.shape)
        w_stp = np.mean(unit_fr * fr_exp_filt, axis=-2)
        fr_tot = np.mean(unit_fr, axis=0)
    return w_stp, fr_tot

        
def get_stp_data(trial_name, tau, lag_range):
    """Load preprocessed ITN firing rate data of given trial name
    tau: exponential filter time constant (second), scalar or array
    lag_range: maximum time lag range (ms)
    """
    PN_spk_file, ITN_fr_file = get_file(trial_name)
    w_stp, fr_tot = PN_stp_weights(PN_spk_file, tau, data='fr')
    ITN_fr = xr.open_dataset(ITN_fr_file)

    lags = np.round(np.array(lag_range) / 1000 * ITN_fr.fs).astype(int)
    lags = np.arange(lags[0], lags[1] + 1)
    t_lags = 1000 / ITN_fr.fs * lags

    T = w_stp.shape[-1] - lags[-1]
    w_stp = np.take(w_stp, range(T), axis=-1)
    fr_tot = fr_tot[:T]
    i_start = ITN_fr.i_start + lags

    lag_fr = []
    for p in ITN_fr.population:
        itn_fr = ITN_fr.spike_rate.sel(population=p).values
        lag_fr.append(np.column_stack([itn_fr[i:i + T] for i in i_start]))
    lag_fr = xr.DataArray(lag_fr, coords={'population': ITN_fr.population, 'time': ITN_fr.time[:T], 'lags': t_lags})
    return w_stp, fr_tot, lag_fr

