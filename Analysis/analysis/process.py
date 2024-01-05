import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss

from build_input import get_populations, get_stim_cycle, T_STOP


def get_stim_windows(on_time, off_time, t_start, t_stop=T_STOP, only_on_time=True):
    """Time windows of stimulus cycles
    only_on_time: whether include only on time in the window or also off time
    Return: 2d-array of time windows, each row is the start/end (sec) of a cycle
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    window =  np.array([0, on_time + (0 if only_on_time else off_time)])
    windows = t_start + window + t_cycle * np.arange(n_cycle)[:, None]
    if windows[-1, 1] > t_stop:
        windows = windows[:-1]
    return windows


def get_stim_cycle_dict(fs, on_time, off_time, t_start, t_stop=T_STOP):
    """Parameters of stimulus cycles"""
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    stim_cycle = dict(
        t_cycle = t_cycle, n_cycle = n_cycle,
        t_start = t_start, on_time = on_time,
        i_start = int(t_start * fs), i_cycle = int(t_cycle * fs)
    )
    return stim_cycle


def get_seg_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None):
    """Convert input time series during stimulus on time into time segments
    x: input 1d-array or 2d-array where time is the last axis
    fs: sampling frequency (Hz)
    on_time, off_time: on / off time durations
    t_start, t: start and stop time of the stimulus cycles
        If t is an array of time points, the last point is used as stop time
    tseg: time segment length. Defaults to on_time if not specified
    Return:
        x_on: same number of dimensions as input, time segments concatenated
        nfft: number of time steps per segment
        stim_cycle: parameters of stimulus cycles
    """
    x = np.asarray(x)
    in_dim = x.ndim
    if in_dim == 1:
        x = x.reshape(1, x.size)
    t = np.asarray(t)
    t_stop = t.size / fs if t.ndim else t
    if tseg is None:
        tseg = on_time # time segment length for PSD (second)
    stim_cycle = get_stim_cycle_dict(fs, on_time, off_time, t_start, t_stop)

    nfft = int(tseg * fs) # steps per segment
    i_on = int(on_time * fs)
    nseg_cycle = int(np.ceil(i_on / nfft))
    x_on = np.zeros((x.shape[0], stim_cycle['n_cycle'] * nseg_cycle * nfft))
    i_start, i_cycle = stim_cycle['i_start'], stim_cycle['i_cycle']

    for i in range(stim_cycle['n_cycle']):
        m = i_start + i * i_cycle
        for j in range(nseg_cycle):
            xx = x[:, m + j * nfft:m + min((j + 1) * nfft, i_on)]
            n = (i * nseg_cycle + j) * nfft
            x_on[:, n:n + xx.shape[1]] = xx
    if in_dim == 1:
        x_on = x_on.ravel()
    return x_on, nfft, stim_cycle


def get_psd_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None, axis=-1):
    x_on, nfft, stim_cycle = get_seg_on_stimulus(
        x, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, pxx = ss.welch(x_on, fs=fs, window='boxcar',
                      nperseg=nfft, noverlap=0, axis=axis)
    return f, pxx, stim_cycle


def get_coh_on_stimulus(x, y, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None):
    xy = np.array([x, y])
    xy_on, nfft, _ = get_seg_on_stimulus(
        xy, fs, on_time, off_time, t_start, t=t, tseg=tseg)
    f, cxy = ss.coherence(xy_on[0], xy_on[1], fs=fs,
        window='boxcar', nperseg=nfft, noverlap=0)
    return f, cxy


def firing_rate(spikes_df, num_cells=None, time_windows=(0.,), frequency=True):
    """
    Count number of spikes for each cell.
    spikes_df: dataframe of node id and spike times (ms)
    num_cells: number of cells (that determines maximum node id)
    time_windows: list of time windows for counting spikes (second)
    frequency: whether return firing frequency in Hz or just number of spikes
    """
    if not spikes_df['timestamps'].is_monotonic:
        spikes_df = spikes_df.sort_values(by='timestamps')
    if num_cells is None:
        num_cells = spikes_df['node_ids'].max() + 1
    time_windows = 1000. * np.asarray(time_windows).ravel()
    if time_windows.size % 2:
        time_windows = np.append(time_windows, spikes_df['timestamps'].max())
    nspk = np.zeros(num_cells, dtype=int)
    n, N = 0, time_windows.size
    count = False
    for t, i in zip(spikes_df['timestamps'], spikes_df['node_ids']):
        while n < N and t > time_windows[n]:
            n += 1
            count = not count
        if count:
            nspk[i] = nspk[i] + 1
    if frequency:
        nspk = nspk / (total_duration(time_windows) / 1000)
    return nspk


def total_duration(time_windows):
    return np.diff(np.reshape(time_windows, (-1, 2)), axis=1).sum()


def pop_spike_rate(spike_times, time=None, time_points=False, frequeny=False):
    """Count spike histogram
    spike_times: spike times
    time: tuple of (start, stop, step)
    time_points: evenly spaced time points. If used, argument `time` is ignored.
    frequeny: whether return spike frequency in Hz or count
    """
    if time_points is None:
        time_points = np.arange(*time)
        dt = time[2]
    else:
        time_points = np.asarray(time_points).ravel()
        dt = (time_points[-1] - time_points[0]) / (time_points.size - 1)
    bins = np.append(time_points, time_points[-1] + dt)
    spike_rate, _ = np.histogram(np.asarray(spike_times), bins)
    if frequeny:
        spike_rate = 1000 / dt * spike_rate
    return spike_rate


def group_spike_rate_to_xarray(spikes_df, time, group_ids,
                               group_dims=['population', 'assembly']):
    """Convert spike times into spike rate of neuron groups in xarray dataset
    spikes_df: dataframe of node ids and spike times
    time: left edges of time bins
    group_ids: dictionary of {group index: group ids}
    group_dims: dimensions in group index. Defaults to ['population', 'assembly']
    """
    time = np.asarray(time)
    fs = 1000 * (time.size - 1) / (time[-1] - time[0])
    if not isinstance(group_dims, list):
        group_dims = [group_dims]
        group_ids = {(k, ): v for k, v in group_ids.items()}
    group_index = pd.MultiIndex.from_tuples(group_ids, names=group_dims)
    grp_rspk = xr.Dataset(
        dict(
            spike_rate = (
                ['group', 'time'],
                [pop_spike_rate(
                    spikes_df.loc[spikes_df['node_ids'].isin(ids), 'timestamps'],
                    time_points=time,  frequeny=True
                ) / len(ids) for ids in group_ids.values()]
            ),
            population_number = ('group', [len(ids) for ids in group_ids.values()])
        ),
        coords = {'group': group_index, 'time': time + 1000 / fs / 2},
        attrs = {'fs': fs}
    ).unstack('group').transpose(*group_dims, 'time')
    return grp_rspk


def combine_spike_rate(grp_rspk, dim, index=slice(None)):
    """Combine spike rate of neuron groups into a new xarray dataset
    grp_rspk: xarray dataset of group spike rate
    dim: group dimension(s) along which to combine
    index: slice or indices of selected groups to combine. Defaults to all
    """
    if not isinstance(dim, list):
        dim = [dim]
        index = [index]
    elif isinstance(index, slice):
        index = [index] * len(dim)
    grp_rspk = grp_rspk.sel(**dict(zip(dim, index)))
    rspk_weighted = grp_rspk.spike_rate.weighted(grp_rspk.population_number)
    combined_rspk = rspk_weighted.mean(dim=dim).to_dataset(name='spike_rate')
    combined_rspk = combined_rspk.assign_attrs(**grp_rspk.attrs).assign(
        population_number = grp_rspk.population_number.sum(dim=dim)
    )
    return combined_rspk


def windowed_xarray(da, windows, dim='time',
                    new_coord_name='cycle', new_coord=None):
    """Divide xarray into windows of equal size along an axis
    da: input DataArray
    windows: 2d-array of windows
    dim: dimension along which to divide
    new_coord_name: name of new dimemsion along which to concatenate windows
    new_coord: pandas Index object of new coordinates. Defaults to integer index
    """
    win_da = [da.sel({dim: slice(*w)}) for w in windows]
    n_win = min(x.coords[dim].size for x in win_da)
    idx = {dim: slice(n_win)}
    coords = da.coords[dim].isel(idx).coords
    win_da = [x.isel(idx).assign_coords(coords) for x in win_da]
    if new_coord is None:
        new_coord = pd.Index(range(len(win_da)), name=new_coord_name)
    win_da = xr.concat(win_da, dim=new_coord)
    return win_da


def group_windows(win_da, win_grp_idx={}, win_dim='cycle'):
    """Group windows into a dictionary of DataArrays
    win_da: input windowed DataArrays
    win_grp_idx: dictionary of {window group id: window indices}
    win_dim: dimension for different windows
    Return: dictionaries of {window group id: DataArray of grouped windows}
        win_on / win_off for windows selected / not selected by `win_grp_idx` 
    """
    win_on, win_off = {}, {}
    for g, w in win_grp_idx.items():
        win_on[g] = win_da.sel({win_dim: w})
        win_off[g] = win_da.drop_sel({win_dim: w})
    return win_on, win_off


def average_group_windows(win_da, win_dim='cycle', grp_dim='unique_cycle'):
    """Average over windows in each group and stack groups in a DataArray
    win_da: input dictionary of {window group id: DataArray of grouped windows}
    win_dim: dimension for different windows
    grp_dim: dimension along which to stack average of window groups 
    """
    win_avg = {g: x.mean(dim=win_dim) for g, x in win_da.items()}
    win_avg = xr.concat(win_avg.values(),
                        dim=pd.Index(win_avg.keys(), name=grp_dim))
    return win_avg

