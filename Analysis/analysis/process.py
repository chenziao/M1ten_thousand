import numpy as np
import pandas as pd
import scipy.signal as ss

from build_input import get_populations, get_stim_cycle, T_STOP


def get_seg_on_stimulus(x, fs, on_time, off_time,
                        t_start, t=T_STOP, tseg=None):
    x = np.asarray(x)
    in_dim = x.ndim
    if in_dim == 1:
        x = x.reshape(1, x.size)
    t = np.asarray(t)
    t_stop = t.size / fs if t.ndim else t
    if tseg is None:
        tseg = on_time # time segment length for PSD (second)
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)

    nfft = int(tseg * fs) # steps per segment
    i_start = int(t_start * fs)
    i_on = int(on_time * fs)
    i_cycle = int(t_cycle * fs)
    nseg_cycle = int(np.ceil(i_on / nfft))
    x_on = np.zeros((x.shape[0], n_cycle * nseg_cycle * nfft))

    for i in range(n_cycle):
        m = i_start + i * i_cycle
        for j in range(nseg_cycle):
            xx = x[:, m + j * nfft:m + min((j + 1) * nfft, i_on)]
            n = (i * nseg_cycle + j) * nfft
            x_on[:, n:n + xx.shape[1]] = xx
    if in_dim == 1:
        x_on = x_on.ravel()

    stim_cycle = {'t_cycle': t_cycle, 'n_cycle': n_cycle,
                  't_start': t_start, 'on_time': on_time,
                  'i_start': i_start, 'i_cycle': i_cycle}
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


def pop_spike_rate(spike_times, time, frequeny=False):
    t = np.arange(*time)
    t = np.append(t, t[-1] + time[2])
    spike_rate, _ = np.histogram(np.asarray(spike_times), t)
    if frequeny:
        spike_rate = 1000 / time[2] * spike_rate
    return spike_rate

