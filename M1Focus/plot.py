import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys
from bmtool.util import util
from build_input import get_populations
from bmtk.simulator import bionet

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster
from bmtk.analyzer.spike_trains import plot_rates_boxplot

CONFIG = "config.json"


def raster(pop_spike, pop_color, s=0.1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for p, spike_df in reversed(pop_spike.items()):
        ax.scatter(spike_df['timestamps'], spike_df['node_ids'],
                   c='tab:' + pop_color[p], s=s, label=p)
    ax.set_title('Spike Raster Plot')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cell ID')
    ax.grid(True)
    return ax


def firing_rate_histogram(pop_fr, pop_color, bins=30, min_fr=None, logscale=False, stacked=True, ax=None):
    if min_fr is not None:
        pop_fr = {p: np.fmax(fr, min_fr) for p, fr in pop_fr.items()}
    fr = np.concatenate(list(pop_fr.values()))
    if logscale:
        fr = fr[np.nonzero(fr)[0]]
        bins = np.geomspace(fr.min(), fr.max(), bins + 1)
    else:
        bins = np.linspace(fr.min(), fr.max(), bins + 1)
    pop_names = list(pop_fr.keys())
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if stacked:
        ax.hist(pop_fr.values(), bins=bins, label=pop_names,
                color=[pop_color[p] for p in pop_names], stacked=True)
    else:
        for p, fr in pop_fr.items():
            ax.hist(fr, bins=bins, label=p, color=pop_color[p], alpha=0.5)
    if logscale:
        ax.set_xscale('log')
        plt.draw()
        xt = ax.get_xticks()
        xtl = [x.get_text() for x in ax.get_xticklabels()]
        xt = np.append(xt, min_fr)
        xtl.append('0')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(loc='upper right')
    ax.set_title('Firing Rate Histogram')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    return ax


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


def plot(choose, spike_file=None, config=None, figsize=(6.4, 4.8)):

    global CONFIG
    if config is None:
        config = CONFIG
    else:
        CONFIG = config
    if spike_file is None:
        spike_file = 'output/spikes.h5'

    pop_color = {'CP': 'blue', 'CS': 'green', 'FSI': 'red', 'LTS': 'purple'}
    pop_names = list(pop_color.keys())

    if choose<=2:
        nodes = util.load_nodes_from_config(config)
        network_name = 'cortex'
        cortex_df = nodes[network_name]

        with h5py.File(spike_file) as f:
            spikes_df = pd.DataFrame({
                'node_ids': f['spikes'][network_name]['node_ids'],
                'timestamps': f['spikes'][network_name]['timestamps']
            })
            spikes_df.sort_values(by='timestamps', inplace=True, ignore_index=True)

    if choose==1:
        spikes_df['pop_name'] = cortex_df.loc[spikes_df['node_ids'], 'pop_name'].values
        pop_spike = get_populations(spikes_df, pop_names)

        print("Plotting cortex spike raster")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        raster(pop_spike, pop_color, ax=ax)
        plt.show()
        #_ = plot_raster(config_file=config, group_by='pop_name')
        return pop_spike

    elif choose==2:
        conf = bionet.Config.from_json(config)
        t_stop = conf['run']['tstop'] / 1000

        frs = firing_rate(spikes_df, num_cells=len(cortex_df), time_windows=(0., t_stop))
        cortex_nodes = get_populations(cortex_df, pop_names, only_id=True)
        pop_fr = {p: frs[nid] for p, nid in cortex_nodes.items()}

        print('Firing rate: mean/(std)')
        for p, fr in pop_fr.items():
            print(f'{p}: {fr.mean():.4g}/({fr.std():.4g})')

        print("Plotting firing rates")
        min_fr = 0.5 / total_duration((0., t_stop)) # to replace 0 spikes
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        firing_rate_histogram(pop_fr, pop_color, bins=20, min_fr=min_fr,
                              logscale=True, stacked=False, ax=ax)
        plt.show()
        return frs

    elif choose==3:
        print("plotting voltage trace")
        _ = plot_traces(config_file=config, report_name='v_report',
                        group_by='pop_name', average=True)

    else:
        print('Choice ID not identified.')


if __name__ == '__main__':
    argv = sys.argv[1:]
    narg = len(argv)
    if narg > 0:
        argv[0] = int(argv[0])
    plot(*argv)