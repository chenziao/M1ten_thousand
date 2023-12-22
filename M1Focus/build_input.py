import os
import csv
import time
import argparse

import numpy as np
from bmtool.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


INPUT_PATH = "./input"
STIMULUS = ['baseline', 'short', 'long']

N_ASSEMBLIES = 10  # number of assemblies
NET_SEED = 4321  # random seed for network r.v.'s (e.g. assemblies, firing rate)
PSG_SEED = 0  # poisson spike generator random seed for different trials
rng = np.random.default_rng(NET_SEED)

T_STOP = 31.  # sec. Simulation time
T_START = 1.0  # sec. Time to start burst input
t_start = T_START
on_time = 1.0  # sec. Burst input duration
off_time = 0.5  # sec. Silence duration

SHELL_FR = {
    'CP': (1.9, 1.8),
    'CS': (1.3, 1.4),
    'FSI': (7.5, 6.0),
    'LTS': (5.0, 5.9)
}  # firing rate of shell neurons (mean, stdev)
SHELL_CONSTANT_FR = False  # whether use constant firing rate for shell neurons


def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))  # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)


def lognormal(mean, stdev, size=None):
    """Generate random values from lognormal given mean and stdev"""
    sigma2 = np.log((stdev / mean) ** 2 + 1)
    mu = np.log(mean) - sigma2 / 2
    sigma = sigma2 ** 0.5
    return rng.lognormal(mu, sigma, size)


def psg_lognormal_fr(psg, node_ids, mean, stdev, times):
    """Generate lognormal distributed firing rate for each node independently.
    Then add the firing rate of each node to the given PoissonSpikeGenerator.
    """
    firing_rates = lognormal(mean, stdev, len(node_ids))
    for node_id, fr in zip(node_ids, firing_rates):
        psg.add(node_ids=node_id, firing_rate=fr, times=times)
    return firing_rates


def df2node_id(df):
    """Get node ids from a node dataframe into a list"""
    return df.index.tolist()


def get_pop(node_df, pop_name):
    """Get nodes with given population name from the nodes dataframe"""
    return node_df.loc[node_df['pop_name'] == pop_name]


def get_pop_id(node_df, pop_name):
    """Get node ids with given population name from the nodes dataframe"""
    return df2node_id(get_pop(node_df, pop_name))


def get_populations(node_df, pop_names, only_id=False):
    """Get node dataframes of multiple populations from the nodes dataframe"""
    func = get_pop_id if only_id else get_pop
    return {p: func(node_df, p) for p in pop_names}


def get_assembly(Thal_nodes, Cortex_nodes, n_assemblies):
    """Divide PNs into n_assemblies and return lists of ids in each assembly"""
    CP_nodes, CS_nodes = Cortex_nodes['CP'], Cortex_nodes['CS']
    num_CP, num_CS = len(CP_nodes), len(CS_nodes)
    num_PN = len(Thal_nodes)
    if num_CP + num_CS != num_PN:
        raise ValueError("Number of thalamus cells don't match number of PNs")

    n_per_assemb = num_prop(np.ones(n_assemblies), num_PN)
    split_idx = np.cumsum(n_per_assemb)[:-1]  # indices at which to split
    assemb_idx = rng.permutation(num_PN)  # random shuffle for assemblies
    assemb_idx = np.split(assemb_idx, split_idx)  # split into assemblies

    Thal_ids = np.array(Thal_nodes)
    PN_ids = np.array(CP_nodes + CS_nodes)
    Thal_assy = []
    PN_assy = []
    for idx in assemb_idx:
        idx = np.sort(idx)
        Thal_assy.append(Thal_ids[idx])
        PN_assy.append(PN_ids[idx])
    return Thal_assy, PN_assy


def get_stim_cycle(on_time=0.5, off_time=0.5, t_start=T_START, t_stop=T_STOP):
    """Get burst input stimulus parameters, (duration, number) of cycles"""
    t_cycle = on_time + off_time
    n_cycle = int(np.floor((t_stop + off_time - t_start) / t_cycle))
    return t_cycle, n_cycle


def get_psg_short(psg, Thal_assy, firing_rate=(0., 0.),
                  on_time=0.5, off_time=0.5, t_start=T_START, t_stop=T_STOP):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Short burst is delivered to each assembly sequentially within each cycle.
    psg: PoissonSpikeGenerator object
    Thal_assy: list of node ids in each assembly
    firing_rate: 2-tuple of firing rate at on and off time, respectively
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    """
    firing_rate = np.asarray(firing_rate)
    if firing_rate.size == 1:
        firing_rate = np.array((firing_rate.item(), 0.))
    else:
        firing_rate = firing_rate[:2]
    if firing_rate.max() <= 0:
        return psg

    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies

    times = np.empty((n_assemblies, n_cycle * 4 + 2))
    times[:, 0] = 0.
    times[:, -1] = t_stop
    for j in range(n_cycle):
        t_cyc = t_start + t_cycle * j
        window = np.linspace(t_cyc, t_cyc + on_time, n_assemblies + 1)
        for i in range(n_assemblies):
            times[i, j * 4 + 1:j * 4 + 5] = np.repeat(window[i:i + 2], 2)
    frs = np.insert(np.tile(firing_rate, n_cycle), 0, firing_rate[1])
    frs = np.repeat(frs, 2)

    for i in range(n_assemblies):
        psg.add(node_ids=Thal_assy[i], firing_rate=frs, times=times[i])
    return psg


def get_psg_long(psg, Thal_assy, firing_rate=(0., 0.),
                 on_time=0.5, off_time=0.5, t_start=T_START, t_stop=T_STOP):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Long burst is delivered to one assembly in each cycle.
    psg: PoissonSpikeGenerator object
    Thal_assy: list of node ids in each assembly
    firing_rate: 2-tuple of firing rate at on and off time, respectively
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    """
    firing_rate = np.asarray(firing_rate)
    if firing_rate.size == 1:
        firing_rate = np.array((firing_rate.item(), 0.))
    else:
        firing_rate = firing_rate[:2]
    if firing_rate.max() <= 0:
        return psg

    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies

    times = [[0] for _ in range(n_assemblies)]
    for j in range(n_cycle):
        t_cycle_start = t_start + t_cycle * j
        window = np.array((t_cycle_start, t_cycle_start + on_time))
        i = j % n_assemblies
        times[i].extend(np.repeat(window, 2))

    frs = []
    for i in range(n_assemblies):
        ts = times[i]
        ts.append(t_stop)
        n = (len(ts) - 2) // 4
        if len(frs) != len(ts):
            frs = np.insert(np.tile(firing_rate, n), 0, firing_rate[1])
            frs = np.repeat(frs, 2)
        psg.add(node_ids=Thal_assy[i], firing_rate=frs, times=ts)
    return psg


def build_input(t_stop=T_STOP, t_start=T_START, n_assemblies=N_ASSEMBLIES,
                psg_seed=PSG_SEED, input_path=INPUT_PATH, stimulus=STIMULUS):
    print("Building all input spike trains...")
    start_timer = time.perf_counter()

    # Get nodes in pandas dataframe
    nodes = util.load_nodes_from_config("config.json")
    pop_names = ['CP', 'CS', 'FSI', 'LTS']
    Cortex_nodes = get_populations(nodes['cortex'], pop_names, only_id=True)

    # Determines node ids for baseline input
    pop_size = [len(n) for n in Cortex_nodes.values()]
    Base_nodes = np.split(df2node_id(nodes['baseline']), np.cumsum(pop_size))
    Base_nodes = dict(zip(pop_names, [n.tolist() for n in Base_nodes[:-1]]))
    baseline_id_file = os.path.join(input_path, "Baseline_ids.csv")
    # rows: baseline ids for each population, cortex ids for each population
    with open(baseline_id_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(Base_nodes.values())
        writer.writerows(Cortex_nodes.values())

    # Assign assemblies for PNs
    Thal_nodes = df2node_id(nodes['thalamus'])
    Thal_assy, PN_assy = get_assembly(Thal_nodes, Cortex_nodes, n_assemblies)
    assembly_id_file = os.path.join(input_path, "Assembly_ids.csv")
    # rows: thalamus ids for each assembly, cortex ids for each assembly
    with open(assembly_id_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(Thal_assy)
        writer.writerows(PN_assy)

    Thal_burst_fr = 50.0  # Hz. Poisson mean firing rate for burst input
    Thal_const_fr = 10.0  # Hz.
    PN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to PNs
    ITN_baseline_fr = 20.0  # Hz. Firing rate for baseline input to ITNs
    sim_time = (0, t_stop)  # Whole simulation

    # Baseline input
    if 'baseline' in stimulus:
        psg = PoissonSpikeGenerator(population='baseline', seed=psg_seed)
        psg.add(node_ids=Base_nodes['CP'] + Base_nodes['CS'],
                firing_rate=PN_baseline_fr, times=sim_time)
        psg.add(node_ids=Base_nodes['FSI'] + Base_nodes['LTS'],
                firing_rate=ITN_baseline_fr, times=sim_time)
        psg.to_sonata(os.path.join(input_path, "baseline.h5"))

    # Constant thalamus input
    if 'const' in stimulus:
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_long(psg, Thal_assy, [Thal_const_fr] * 2,
                           on_time, off_time, t_start=t_start, t_stop=t_stop)
        psg.to_sonata(os.path.join(input_path, "thalamus_const.h5"))

    # Short burst thalamus input
    if 'short' in stimulus:
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_short(psg, Thal_assy, Thal_burst_fr, on_time, off_time,
                            t_start=t_start, t_stop=t_stop)
        psg.to_sonata(os.path.join(input_path, "thalamus_short.h5"))

    # Long burst thalamus input
    if 'long' in stimulus:
        psg = PoissonSpikeGenerator(population='thalamus', seed=psg_seed + 100)
        psg = get_psg_long(psg, Thal_assy, Thal_burst_fr, on_time, off_time,
                           t_start=t_start, t_stop=t_stop)
        psg.to_sonata(os.path.join(input_path, "thalamus_long.h5"))

    print("Core cells: %.3f sec" % (time.perf_counter() - start_timer))

    # These inputs are for the baseline firing rates of the cells in the shell.
    if 'shell' in nodes:
        start_timer = time.perf_counter()

        # Generate Poisson spike trains for shell cells
        psg = PoissonSpikeGenerator(population='shell', seed=psg_seed + 1000)
        shell_nodes = get_populations(nodes['shell'], pop_names, only_id=True)

        # Select effective nodes in shell that only has connections to core
        edge_paths = util.load_config("config.json")['networks']['edges']
        for path in edge_paths:
            if 'shell_cortex' in path['edge_types_file']:
                _, shell_edges = util.load_edges(**path)
        effective_shell = set(shell_edges['source_node_id'])

        def effective_cell(cell_list, effect_set):
            effect_list = [x for x in cell_list if x in effect_set]
            ratio = len(effect_list) / len(cell_list)
            return effect_list, ratio

        print("Proportion of effective cells in shell.")

        fr_list = []
        for p, node_ids in shell_nodes.items():
            node_ids, ratio = effective_cell(node_ids, effective_shell)
            print("%.1f%% effective %s." % (100 * ratio, p))

            if SHELL_CONSTANT_FR:
                # Constant mean firing rate for all cells
                fr = SHELL_FR[p][0]
                psg.add(node_ids=node_ids, firing_rate=fr, times=sim_time)
            else:
                # Lognormal distributed mean firing rate
                fr = psg_lognormal_fr(psg, node_ids, mean=SHELL_FR[p][0],
                                      stdev=SHELL_FR[p][1], times=sim_time)
                fr_list.append(fr)

        if not SHELL_CONSTANT_FR:
            fr_file = os.path.join(input_path, "Lognormal_FR.csv")
            with open(fr_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(fr_list)

        psg.to_sonata(os.path.join(input_path, "shell.h5"))
        print("Shell cells: %.3f sec" % (time.perf_counter() - start_timer))

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-stop', '--t_stop', type=float,
                        nargs='?', default=T_STOP,
                        help="Simulation stop time", metavar='t_stop')
    parser.add_argument('-start', '--t_start', type=float,
                        nargs='?', default=T_START,
                        help="Simulation start period", metavar='t_start')
    parser.add_argument('-n', '--n_assemblies', type=int,
                        nargs='?', default=N_ASSEMBLIES,
                        help="Number of assemblies", metavar='# Assemblies')
    parser.add_argument('-net', '--net_seed', type=int,
                        nargs='?', default=NET_SEED,
                        help="Network random seed", metavar='Network Seed')
    parser.add_argument('-psg', '--psg_seed', type=int,
                        nargs='?', default=PSG_SEED,
                        help="Poisson generator seed", metavar='PSG Seed')
    parser.add_argument('-path', '--input_path', type=str,
                        nargs='?', default=INPUT_PATH,
                        help="Input path", metavar='Input Path')
    parser.add_argument('-s', '--stimulus', type=str,
                        nargs="*", default=STIMULUS,
                        help="List of stimulus types", metavar='Stimulus')
    args = parser.parse_args()

    rng = np.random.default_rng(args.net_seed)
    build_input(t_stop=args.t_stop, t_start=args.t_start,
                n_assemblies=args.n_assemblies, psg_seed=args.psg_seed,
                input_path=args.input_path, stimulus=args.stimulus)
