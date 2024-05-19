import numpy as np
import pandas as pd
import xarray as xr
import os
import json
import warnings

from bmtool.util.util import load_nodes_from_paths
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from functools import partial

from analysis import  plot, utils, process
from build_input import get_populations, t_start


RESULT_PATH = "simulation_results"
OUTPUT_PATH = "analysis_results"

network_name = 'cortex'
PN_pop_names = ['CP', 'CS']
ITN_pop_names = ['FSI', 'LTS']
T_START = t_start
# cross correlation
pop_groups = {'PN': PN_pop_names, 'ITN': ITN_pop_names}
elec_id = 0

def set_variables(**kwargs):
    global_vars = globals()
    for key, value in kwargs.items():
        global_vars[key] = value
    # set up data path
    global PN_SPK_PATH, ITN_FR_PATH, TSPK_PATH, BIN_SPK_PATH, XCORR_PATH
    PN_SPK_PATH = os.path.join(OUTPUT_PATH, 'PN_spikes')
    ITN_FR_PATH = os.path.join(OUTPUT_PATH, 'ITN_fr')
    TSPK_PATH = os.path.join(OUTPUT_PATH, 'spike_times')
    BIN_SPK_PATH = os.path.join(OUTPUT_PATH, 'binned_spikes')
    XCORR_PATH = os.path.join(OUTPUT_PATH, 'wave_fr_xcorr')
    if not os.path.isdir(PN_SPK_PATH):
        os.mkdir(PN_SPK_PATH)
    if not os.path.isdir(ITN_FR_PATH):
        os.mkdir(ITN_FR_PATH)
    if not os.path.isdir(TSPK_PATH):
        os.mkdir(TSPK_PATH)
    if not os.path.isdir(BIN_SPK_PATH):
        os.mkdir(BIN_SPK_PATH)
    if not os.path.isdir(XCORR_PATH):
        os.mkdir(XCORR_PATH)

set_variables()


def get_trials(filter=[], trials=None, revert_junction=False, exclude=None, exclude_kwargs={}):
    """Get name of trials from simulation result directory
    filter: str or nested list of str. Get trials with matching substrings
        If it is nested list, apply intersection to the conditions in the inner
        lists and then union to the outer list.
        E.g., [('a', 'b'), ('c', 'd'), 'e'], means (a AND b) OR (c AND d) OR e.
    trials: list of trial names. If not specified, obtain from result directory
    revert_junction: whether revert the junction relation in the filter, i.e.,
        apply union to the inner lists and then intersection to the outer list
    exclude: if specified, used as `filter` to obtain trials need to be excluded
    exclude_kwargs: additional keyword arguments for exclude
    """
    if trials is None:
        trials = [f for f in os.listdir(RESULT_PATH) if f[0] != '.']
    if isinstance(filter, str):
        filter = [[filter]]
    elif len(filter) == 0:
        filter = [filter]
    if revert_junction:
        inner = any
        outer = set.intersection
        filt = set(trials)
    else:
        inner = all
        outer = set.union
        filt = set()
    for ft in filter:
        if isinstance(ft, str):
            ft = [ft]
        filt = outer(filt, set(f for f in trials if inner(s in f for s in ft)))
    if exclude is not None:
        filt = filt - set(get_trials(filter=exclude, **exclude_kwargs))
    trials = [f for f in trials if f in filt]
    return trials


def get_file(trial_name):
    PN_spk_file = os.path.join(PN_SPK_PATH, trial_name + '.npz')
    ITN_fr_file = os.path.join(ITN_FR_PATH, trial_name + '.nc')
    return PN_spk_file, ITN_fr_file


def load_trial(trial_name, pop_names):
    # Load trial information
    trial_path = os.path.join(RESULT_PATH, trial_name)
    _, paths, stim_info, _ = utils.get_trial_info(trial_path)
    _, NODE_FILES, SPIKE_FILE = paths
    t_stop, _, stim_params = stim_info
    t_start = stim_params.get('t_start', T_START)
    # Load trial data
    node_df = load_nodes_from_paths(NODE_FILES)[network_name]
    pop_ids = get_populations(node_df, pop_names, only_id=True)
    spikes_df = utils.load_spikes_to_df(SPIKE_FILE, network_name)
    return pop_ids, spikes_df, t_start, t_stop


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
    pop_names = PN_pop_names * process_PN + ITN_pop_names * process_ITN
    # Load trial info and data
    pop_ids, spikes_df, t_start, t_stop = load_trial(trial_name, pop_names)

    # Parameters
    time_ct_edge = np.linspace(0, 1000 * t_stop, int(t_stop * fs_ct), endpoint=False)
    time_ct = time_ct_edge + 1000 / fs_ct / 2
    i_start_ct = pd.Index(time_ct).get_indexer([1000 * t_start], method='bfill')[0]

    time_fr = np.linspace(0, 1000 * t_stop, int(t_stop * fs_fr), endpoint=False)
    time_fr = time_fr + 1000 / fs_fr / 2
    time_fr = time_fr[(time_fr >= time_ct[0]) & (time_fr <= time_ct[-1])]
    i_start_fr = pd.Index(time_fr).get_indexer([1000 * t_start], method='bfill')[0]

    # PN units spike counts and firing rate
    if process_PN:
        PN_node_ids = np.concatenate([pop_ids[p] for p in PN_pop_names])
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
        ITN_rspk = ITN_rspk.assign_attrs(gauss_filt_sigma=filt_sigma,
            fs=fs_ct, t_start=t_start, i_start=i_start_ct)
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


def get_ITN_data(trial_name, lag_range):
    PN_spk_file, ITN_fr_file = get_file(trial_name)
    with np.load(PN_spk_file) as f:
        fs = f['fs_fr']
        time_fr = f['time_fr']
        i_start = f['i_start_fr']
        attrs = dict(fs=fs, i_start=i_start)
    lags = np.round(np.array(lag_range) / 1000 * fs).astype(int)
    lags = np.arange(lags[0], lags[1] + 1)
    t_lags = 1000 / fs * lags
    i_start = i_start + lags
    T = time_fr.size - i_start[-1]

    ITN_fr = xr.open_dataset(ITN_fr_file).spike_rate
    coords, dims = {**ITN_fr.coords, 'time': time_fr}, ITN_fr.dims
    ITN_fr = xr.DataArray(interp1d(ITN_fr.time, ITN_fr, axis=dims.index('time'),
        assume_sorted=True)(time_fr), coords=coords, dims=dims)
    lag_fr = []
    for p in ITN_fr.population:
        itn_fr = ITN_fr.sel(population=p).values
        lag_fr.append(np.column_stack([itn_fr[i:i + T] for i in i_start]))
    lag_fr = xr.DataArray(lag_fr, name='lagged firing rate', coords=dict(
        population=ITN_fr.population, time=time_fr[:T], lags=t_lags))
    return lag_fr, T, attrs


def get_stp_data(trial_name, tau, lag_range, data='fr'):
    """Load preprocessed ITN firing rate data of given trial name
    tau: exponential filter time constant (second), scalar or array
    lag_range: maximum time lag range (ms)
    """
    lag_fr, T, _ = get_ITN_data(trial_name, lag_range)
    PN_spk_file, _ = get_file(trial_name)
    w_stp, fr_tot = PN_stp_weights(PN_spk_file, tau, data=data)
    w_stp = np.take(w_stp, range(T), axis=-1)
    fr_tot = fr_tot[:T]
    return w_stp, fr_tot, lag_fr


nid_tspk_to_lil = process.nid_tspk_to_lil

class STP_Jump(object):
    """Class for simulating STP dynamic as jump process"""
    def __init__(self, **params):
        # set parameters
        self.params = ('tau_f', 'tau_d', 'U')
        for p in self.params:
            setattr(self, p, params.get(p))
        # set state variables
        vars = ('t', 'dt', 'u', 'R', 'P')
        self.var_idx = {var: i for i, var in enumerate(vars)}
        for var in vars:
            setattr(self, var, partial(self.get_state, var=var))
        self.states = None

    def set_params(self, tau_f=None, tau_d=None, U=None, **kwargs):
        """Set dynamic parameters"""
        if tau_f is not None:
            self.tau_f = max(tau_f, 0.)
        if tau_d is not None:
            self.tau_d = max(tau_d, 0.)
        if U is not None:
            assert(U > 0 and U <= 1)
            self.U = U
            # set u0, P0 if using u0=U by default
            iu = self.var_idx['u']
            if self.states is not None and self.states[0][iu, 0] < 0:
                iP = self.var_idx['P']
                P = U * self.states[0][self.var_idx['R'], 0] 
                for states in self.states:
                    states[iu, 0] = U
                    states[iP, 0] = P
        notspecified = [p for p in self.params if getattr(self, p) is None]
        if notspecified:
            raise ValueError('Parameters ' + ', '.join(notspecified) + ' not specified')
        tau_f, tau_d, U = self.tau_f, self.tau_d, self.U
        if tau_f:
            Uc = 1 - U
            def update_u(dt, u):
                return U + Uc * (u * np.exp(-dt / tau_f))
        else:
            update_u = lambda *_: U
        self.update_u = update_u
        if tau_d:
            def update_R(dt, R, P):
                return 1 - (1 - (R - P)) * np.exp(-dt / tau_d)
        else:
            update_R = lambda *_: 1.
        self.update_R = update_R

    def update(self, dt, u, R, P):
        """Update u after jump, R before jump, and P between the two jumps"""
        # positional arguments need to follow the order in self.var_idx
        u = self.update_u(dt, u)
        R = self.update_R(dt, R, P)
        P = u * R
        return u, R, P

    def initialize(self, tspk, nid=None, N=None, assume_sorted=True,
                   u0=None, R0=1.0, t0=0.):
        """Set up spike times of the units
        tspk: if `nid` not specified, `tspk` is a list of spike trains of N
            units, where each spike train is a list of spike times (assume sorted)
        nid: if spicified, `tspk` is the list of spike times of all units, `nid`
            is the list of unit ids at corresponding spike times, with same size
            as `tspk`. The ids in `nid` should range from 0 to N - 1, where the
            number of units N 
        N: number of units. If not spcified, is inferred from the maximum value
            of `nid` or from the len of `tspk` if `nid` not specified.
        assume_sorted: whether assume tspk is already sorted
        u0, R0, t0: variables for initial spike before start time
        """
        assert(len(tspk) > 0)
        if nid is None:
            self.N = len(tspk) if N is None else N
            if N is not None:
                tspk = tspk[:N]
                if not assume_sorted:
                    tspk = [sorted(t) for t in tspk]
        else:
            self.N = max(nid) + 1 if N is None else N
            if not assume_sorted:
                nid, tspk = zip(*sorted(zip(nid, tspk)))
            tspk = nid_tspk_to_lil(nid, tspk, self.N)
        if self.states is None:
            # initialize state variables
            if u0 is None:
                u0 = -1 if self.U is None else self.U 
            else:
                assert(u0 >= 0 and u0 <= 1)
            assert(R0 >= 0 and R0 <= 1 and t0 <= 0)
            state0 = dict(t=t0, dt=0., u=u0, R=R0, P=u0 * R0)
            vi = self.var_idx
            it, idt = vi['t'], vi['dt']
            state0 = [state0[var] for var in vi]
            self.states = []
            for ts in tspk:
                states = np.empty((len(vi), len(ts) + 1))
                states[it, 1:] = ts
                states[:, 0] = state0
                states[idt, 1:] = np.diff(states[it])
                self.states.append(states)
        return self

    def simulate(self, tspk=None, nid=None, N=None, assume_sorted=True,
                 u0=None, R0=1.0, t0=0., **params):
        """Simulate STP dynamic as jump process"""
        if tspk is not None:
            self.initialize(tspk, nid=nid, N=N, assume_sorted=assume_sorted,
                            u0=u0, R0=R0, t0=t0)
        params = {**{p: getattr(self, p, None) for p in self.params}, **params}
        self.set_params(**params)
        for states in self.states:
            for i in range(states.shape[1] - 1):
                states[2:, i + 1] = self.update(*states[1:, i])
        return self

    def get_state(self, nid=None, var='P', concat=False):
        """Get state variables"""
        idx = self.var_idx[var]
        nid = np.arange(self.N) if nid is None else np.asarray(nid)
        states = [self.states[i][idx, 1:] for i in nid.ravel()]
        if concat or nid.ndim == 0:
            states = np.concatenate(states)
        return states


class Simple_Jump(object):
    """Class for simulating simplified jump process to approximate STP dynamic"""
    def __init__(self, **params):
        # set parameters
        self.params = ('tau_f', 'tau_d')
        for p in self.params:
            setattr(self, p, params.get(p))
        # set state variables
        vars = ('t', 'dt', 'sf', 'sd')
        self.var_idx = {var: i for i, var in enumerate(vars)}
        for var in vars:
            setattr(self, var, partial(self.get_state, var=var))
        # set estimated variables
        ests = ('P', 'u', 'R')
        for var in ests:
            setattr(self, var, partial(self.get_est, var=var))
        self.states = None

    def set_params(self, tau_f=None, tau_d=None, **kwargs):
        """Set dynamic parameters"""
        if tau_f is not None:
            self.tau_f = max(tau_f, 0.)
        if tau_d is not None:
            self.tau_d = max(tau_d, 0.)
        notspecified = [p for p in self.params if getattr(self, p) is None]
        if notspecified:
            raise ValueError('Parameters ' + ', '.join(notspecified) + ' not specified')
        self.update_f = self.get_update_s(self.tau_f)
        self.update_d = self.get_update_s(self.tau_d)

    @staticmethod
    def get_update_s(tau):
        if tau:
            def update_s(dt, s):
                return 1 + s * np.exp(-dt / tau)
        else:
            update_s = lambda *_: 1.
        return update_s

    def update(self, dt, sf, sd):
        """Update u after jump, R before jump, and P between the two jumps"""
        # positional arguments need to follow the order in self.var_idx
        sf = self.update_f(dt, sf)
        sd = self.update_d(dt, sd)
        return sf, sd

    def initialize(self, tspk, nid=None, N=None, assume_sorted=True,
                   s0=1., t0=0.):
        """Set up spike times of the units
        tspk: if `nid` not specified, `tspk` is a list of spike trains of N
            units, where each spike train is a list of spike times (assume sorted)
        nid: if spicified, `tspk` is the list of spike times of all units, `nid`
            is the list of unit ids at corresponding spike times, with same size
            as `tspk`. The ids in `nid` should range from 0 to N - 1, where the
            number of units N 
        N: number of units. If not spcified, is inferred from the maximum value
            of `nid` or from the len of `tspk` if `nid` not specified.
        assume_sorted: whether assume tspk is already sorted
        s0, t0: variables for initial spike before start time
        """
        assert(len(tspk) > 0)
        if nid is None:
            self.N = len(tspk) if N is None else N
            if N is not None:
                tspk = tspk[:N]
                if not assume_sorted:
                    tspk = [sorted(t) for t in tspk]
        else:
            self.N = max(nid) + 1 if N is None else N
            if not assume_sorted:
                nid, tspk = zip(*sorted(zip(nid, tspk)))
            tspk = nid_tspk_to_lil(nid, tspk, self.N)
        if self.states is None:
            # initialize state variables
            assert(s0 >= 1 and t0 <= 0)
            state0 = dict(t=t0, dt=0., sf=s0, sd=s0)
            vi = self.var_idx
            it, idt = vi['t'], vi['dt']
            state0 = [state0[var] for var in vi]
            self.states = []
            for ts in tspk:
                states = np.empty((len(vi), len(ts) + 1))
                states[it, 1:] = ts
                states[:, 0] = state0
                states[idt, 1:] = np.diff(states[it])
                self.states.append(states)
        return self

    def simulate(self, tspk=None, nid=None, N=None, assume_sorted=True,
                 s0=1., t0=0., **params):
        """Simulate STP dynamic as simplified jump process"""
        if tspk is not None:
            self.initialize(tspk, nid=nid, N=N, assume_sorted=assume_sorted,
                            s0=s0, t0=t0)
        params = {**{p: getattr(self, p, None) for p in self.params}, **params}
        self.set_params(**params)
        for states in self.states:
            for i in range(states.shape[1] - 1):
                states[2:, i + 1] = self.update(*states[1:, i])
        return self

    def get_state(self, nid=None, var='P', concat=False):
        """Get state variables"""
        idx = self.var_idx[var]
        nid = np.arange(self.N) if nid is None else np.asarray(nid)
        states = [self.states[i][idx, 1:] for i in nid.ravel()]
        if concat or nid.ndim == 0:
            states = np.concatenate(states)
        return states

    def run_est(self, U, return_uR=False):
        """Estimate P given U after simulation is done"""
        if self.states is None:
            raise ValueError('Simulation has not run yet.')
        assert(U > 0 and U <= 1)
        Ur = (1 - U) / U
        if self.tau_f and Ur:
            def F(sf_cur, sf_pre):
                return 1 + (sf_cur - 1) / (1 + sf_pre / Ur)
        else: 
            F = lambda *_: 1.
        if self.tau_d:
            if self.tau_f:
                def D(sd_cur, sd_pre, sf_pre):
                    return 1 - (sd_cur - 1 ) / (sd_pre + Ur / sf_pre)
            else:
                def D(sd_cur, sd_pre, sf_pre):
                    return 1 - (sd_cur - 1 ) / (sd_pre + Ur)
        else:
            D = lambda *_: 1.
        i_f = self.var_idx['sf']
        i_d = self.var_idx['sd']
        P = []
        if return_uR:
            self._u, self._R = u, R = [], []
            def est(sf, sd, n):
                global P_
                u_, R_, P_ = (np.empty(n) for _ in range(3))
                u_[:] = U * F(sf[1:], sf[:-1])
                R_[:] = D(sd[1:], sd[:-1], sf[:-1])
                u.append(u_)
                R.append(R_)
                return u_, R_
        else:
            self._u, self._R = None, None
            def est(sf, sd, n):
                global P_
                P_ = np.empty(n)
                u_ = U * F(sf[1:], sf[:-1])
                R_ = D(sd[1:], sd[:-1], sf[:-1])
                return u_, R_
        for states in self.states:
            u_, R_ = est(states[i_f, :], states[i_d, :], states.shape[1] - 1)
            P_[:] = u_ * R_
            P.append(P_)
        self._P = P
        self.U = U

    def get_est(self, nid=None, var='P', concat=False):
        """Get estimated variables"""
        dat = getattr(self, '_' + var, None)
        if dat is None:
            est = None
            print("Warning: " + var + " was not estimated")
        else:
            nid = np.arange(self.N) if nid is None else np.asarray(nid)
            est = [dat[i] for i in nid.ravel()]
            if concat or nid.ndim == 0:
                est = np.concatenate(est)
        return est


def get_pop_stp_info(pre_pop_names, ITN_pop_names):
    # Get synapse parameters
    syn_types = {(p, i): p + '2' + i for p in pre_pop_names for i in ITN_pop_names}
    _, _, _, config_hp = utils.get_trial_info(os.path.join(RESULT_PATH, 'baseline'))
    SYN_PATH = config_hp.get_attr('components', 'synaptic_models_dir')

    syn_params = {}
    for syn, syn_file in syn_types.items():
        with open(os.path.join(SYN_PATH, syn_file + '.json'), 'r') as f:
            syn_p = json.load(f)
        syn_params[syn] = dict(
            tau_f=syn_p['Fac'], tau_d=syn_p['Dep'], U=syn_p['Use'])

    # Get node ids of presynaptic cell types
    pre_ids, _, _, _ = load_trial('baseline', pre_pop_names)
    pre_ids_idx = {}
    for p, ids in pre_ids.items():
        ids = sorted(ids)
        pre_ids_idx[p] = pd.DataFrame(dict(index=range(len(ids))), index=pd.Series(ids, name='node_id'))
    return syn_types, syn_params, pre_ids_idx


def setup_pop_stp(trial_name, pop_stp_info, attrs, overwrite=False):
    syn_types, syn_params, pre_ids_idx = pop_stp_info
    tspk_files = {}
    bin_spk_files = {}
    writefile = {'tspk': {}, 'bin_spk': {}}
    for syn, syn_type in syn_types.items():
        tspk_files[syn] = os.path.join(TSPK_PATH, trial_name + '_' + syn_type + '.npz')
        writefile['tspk'][syn] = overwrite or not os.path.isfile(tspk_files[syn])
        pre = syn[0]
        bin_spk_files[pre] = os.path.join(BIN_SPK_PATH, trial_name + '_' + pre + '.npz')
        writefile['bin_spk'][pre] = overwrite or not os.path.isfile(bin_spk_files[pre])
    process_tspk = any(writefile['tspk'].values())
    process_bin_spk = any(writefile['bin_spk'].values())

    # load spike data
    if process_tspk or process_bin_spk:
        _, spk_df, t_start, t_stop = load_trial(trial_name, pre_ids_idx)
        spk_df = spk_df.set_index('node_ids').rename(columns={'timestamps': 'tspk'})

    # get spike times for each cell type
    if process_tspk:
        pop_tspk = {}
        for p, idx in pre_ids_idx.items():
            ids = idx.index.intersection(spk_df.index)
            tspk = spk_df.loc[ids]
            tspk['nid'] = idx.loc[tspk.index, 'index']
            pop_tspk[p] = tspk

    # create stp jump process objects
    stp_objs = {}
    for syn in syn_types:
        file = tspk_files[syn]
        stp_params = syn_params[syn]
        stp_jump = STP_Jump()
        if writefile['tspk'][syn]:
            pre = syn[0]
            tspk, nid = pop_tspk[pre]['tspk'], pop_tspk[pre]['nid']
            stp_jump.initialize(tspk=tspk, nid=nid, N=pre_ids_idx[pre].size, assume_sorted=False)
            save_lil(file, lil={'tspk': stp_jump.t()}, add_vars=stp_params)
        else:
            tspk, params = load_lil(file)
            if any(stp_params[key] != val for key, val in params.items()):
                print("Warning: using different STP parameters")
                print(params)
                stp_params.update(params)
            stp_jump.initialize(tspk=tspk['tspk'], assume_sorted=True)
        stp_jump.set_params(**stp_params)
        stp_objs[syn] = stp_jump

    # bin spike times
    if process_bin_spk:
        fs = attrs['fs']
        i_start = attrs['i_start']
        time_edge = np.linspace(0, 1000 * t_stop, int(t_stop * fs), endpoint=False)
    bin_spk_idx = {}
    for syn, stp_jump in stp_objs.items():
        pre = syn[0]
        if pre not in bin_spk_idx:
            file = bin_spk_files[pre]
            if writefile['bin_spk'][pre]:
                t = stp_jump.t(concat=True)
                bin_spk = bin_spike_indices(t, time_edge)[i_start:]
                save_lil(file, lil={'bin_spk': bin_spk})
            else:
                bin_spk = load_lil(file)[0]['bin_spk']
            bin_spk_idx[pre] = bin_spk
    return stp_objs, bin_spk_idx


def bin_spike_indices(tspk, time_edge):
    """Bin spike times and return indices of spikes in each bin
    tspk: spike times
    time_edge: left edges of time bins
    """
    bin_idx = np.digitize(tspk, time_edge) - 1  # bin index of each spike
    spk_idx = np.arange(bin_idx.size)  # spike indices
    S = csr_matrix((spk_idx, [bin_idx, spk_idx]), shape=(time_edge.size, spk_idx.size))
    return np.split(S.data, S.indptr[1:-1])


def save_lil(file, lil={}, add_vars={}):
    """Save row-based List of Lists array"""
    xs, idx = {}, {}
    for key, x in lil.items():
        xs[key] = np.concatenate(x)
        idx[key + '_idx'] = np.insert(np.cumsum(list(map(len, x))), 0, 0)
    np.savez_compressed(file, **xs, **idx, **add_vars)


def load_lil(file):
    lil, add_vars = {}, {}
    with np.load(file) as f:
        for key, val in f.items():
            if key.endswith('_idx'):
                key = key.replace('_idx', '')
                lil[key] = val
            else:
                add_vars[key] = val
    for key, idx in lil.items():
        x = add_vars.pop(key)
        lil[key] = [x[i:j] for i, j in zip(idx[:-1], idx[1:])]
    return lil, add_vars


def simulate_stp(stp_objs, **syn_params):
    """Run simulate for all stp objects in stp_objs with parameters in syn_params"""
    for stp_jump in stp_objs:
        stp_jump.simulate(**syn_params)


def get_wave_fr_xcorr(trial_name, wave_kwargs, normalize_lfp=True, normalize_fr=False, max_lag=300., overwrite=False):
    xcorr_file = os.path.join(XCORR_PATH, trial_name + '.nc')
    if overwrite or not os.path.isfile(xcorr_file):
        # Load ITN firing rate data
        _, ITN_fr_file = get_file(trial_name)
        ITN_fr = xr.open_dataset(ITN_fr_file)
        fs, time, i_start = ITN_fr.fs, ITN_fr.time, ITN_fr.i_start
        dt = 1000 / fs
        if normalize_lfp or normalize_fr:
            filt_sigma = ITN_fr.gauss_filt_sigma / dt
        # Get population firing rate
        pop_ids, spikes_df, t_start, t_stop = load_trial(trial_name, sum(pop_groups.values(), []))
        pop_ids = {grp: sum([pop_ids[p] for p in pop], []) for grp, pop in pop_groups.items()}
        time_edge = time - dt / 2
        pop_rspk = process.group_spike_rate_to_xarray(spikes_df, time_edge, pop_ids, group_dims='population')
        pop_waves = process.get_waves(pop_rspk.spike_rate, fs=fs, **wave_kwargs)
        if normalize_fr:
            # Gaussian smoothing
            axis = pop_rspk.spike_rate.dims.index('time')
            sigma = np.zeros(pop_rspk.spike_rate.ndim)
            sigma[axis] = filt_sigma
            pop_rspk.update(dict(smoothed_spike_rate=xr.zeros_like(pop_rspk.spike_rate)))
            pop_rspk.smoothed_spike_rate[:] = gaussian_filter(pop_rspk.spike_rate, sigma)
            pop_waves /= pop_rspk.smoothed_spike_rate
        # Concatenate with waves in LFP
        lfp_file = os.path.join(RESULT_PATH, trial_name, 'ecp.h5')
        lfp = utils.load_ecp_to_xarray(lfp_file).sel(channel_id=elec_id)
        lfp = interp1d(lfp.time, lfp, assume_sorted=True)(time)
        if normalize_lfp:
            lfp /= gaussian_filter(lfp * lfp, filt_sigma) ** 0.5  # normalize by total power
        lfp_waves = process.get_waves(xr.DataArray(lfp, coords={'time': time}), fs=fs, **wave_kwargs)
        pop_waves = xr.concat([pop_waves, lfp_waves.expand_dims(dim={'population': ['LFP']})], dim='population')

        # collect cross correlations into xarray dataset
        wave_pop, waves = pop_waves.population, pop_waves.wave
        t_slice = slice(i_start, None)
        wave_fr_xcorr = []
        for wp in wave_pop:
            for w in waves:
                w_da = pop_waves.sel(population=wp, wave=w)
                for p in ITN_pop_names:
                    xcorr, xcorr_lags = plot.xcorr_coeff(w_da.isel(time=t_slice),
                        ITN_fr.spike_rate.sel(population=p).isel(time=t_slice),
                        max_lag=max_lag, dt=dt, plot=False)
                    wave_fr_xcorr.append(xcorr)

        coords = {'wave_pop': wave_pop.values, 'wave': waves.values, 'FR_population': ITN_pop_names, 'lags': xcorr_lags}
        wave_fr_xcorr = np.array(wave_fr_xcorr).reshape(wave_pop.size, waves.size, len(ITN_pop_names), -1)
        wave_fr_xcorr = xr.Dataset(dict(xcorr=(coords, wave_fr_xcorr)), coords=coords,
            attrs=dict(max_lag=max_lag, n_lags=xcorr_lags.size // 2, fs=fs, duration=t_stop - t_start))
        # Save dataset to file
        wave_fr_xcorr.to_netcdf(xcorr_file)
    else:
        wave_fr_xcorr = xr.open_dataset(xcorr_file)
    return wave_fr_xcorr

