import numpy as np
from bmtool.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

import sys
import os
import csv
import time

randseed = 4321
rng = np.random.default_rng(randseed)

T_SIM = 16.  # sec. Simulation time
N_ASSEMBLIES = 10  # number of assemblies
INPUT_PATH = "./input"


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


def get_pop(node_df, pop_name):
    """Get nodes with given pop_name from the nodes dataframe"""
    return node_df.loc[node_df['pop_name'] == pop_name]


def df2node_id(df):
    """Get node ids from a node dataframe into a list"""
    return df.index.tolist()


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

    Thal_ids = np.array(df2node_id(Thal_nodes))
    PN_ids = np.array(df2node_id(CP_nodes) + df2node_id(CS_nodes))
    Thal_assy = []
    PN_assy = []
    for idx in assemb_idx:
        idx = np.sort(idx)
        Thal_assy.append(Thal_ids[idx])
        PN_assy.append(PN_ids[idx])
    return Thal_assy, PN_assy


def get_stim_cycle(on_time=1.0, off_time=0.5, t_start=0., t_stop=T_SIM):
    """Get burst input stimulus parameters, (duration, number) of cycles"""
    t_cycle = on_time + off_time
    n_cycle = int(np.floor((t_stop + off_time - t_start) / t_cycle))
    return t_cycle, n_cycle


def get_psg_short(Thal_assy, firing_rate=(0., 0.), on_time=1.0, off_time=0.5,
                  t_start=0., t_stop=T_SIM):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Short burst is delivered to each assembly sequentially within each cycle.
    Thal_assy: list of node ids in each assembly
    firing_rate: 2-tuple of firing rate at on and off time, respectively
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies
    psg = PoissonSpikeGenerator(population='thalamus')

    firing_rate = np.asarray(firing_rate) 
    if np.isscalar(firing_rate):
        firing_rate = np.array((firing_rate, 0.))
    else:
        firing_rate = firing_rate[:2]
    if firing_rate.max() <= 0:
        return psg

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


def get_psg_long(Thal_assy, firing_rate=(0., 0.), on_time=1.0, off_time=0.5,
                 t_start=0., t_stop=T_SIM):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Long burst is delivered to one assembly in each cycle.
    Thal_assy: list of node ids in each assembly
    firing_rate: 2-tuple of firing rate at on and off time, respectively
    on_time, off_time: on / off time durations
    t_start, t_stop: start and stop time of the stimulus cycles
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies
    psg = PoissonSpikeGenerator(population='thalamus')

    firing_rate = np.asarray(firing_rate) 
    if np.isscalar(firing_rate):
        firing_rate = np.array((firing_rate, 0.))
    else:
        firing_rate = firing_rate[:2]
    if firing_rate.max() <= 0:
        return psg

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


def build_input(t_stop=T_SIM, n_assemblies=N_ASSEMBLIES):
    print("Building all input spike trains...")
    start_timer = time.perf_counter()

    # Get nodes in pandas dataframe
    nodes = util.load_nodes_from_config("config.json")
    pop_names = ['CP', 'CS', 'FSI', 'LTS']
    Cortex_nodes = {p: get_pop(nodes['cortex'], p) for p in pop_names}

    Thal_nodes = nodes['thalamus']
    Thal_assy, PN_assy = get_assembly(Thal_nodes, Cortex_nodes, n_assemblies)
    assembly_id_file = os.path.join(INPUT_PATH, "Assembly_ids.csv")
    with open(assembly_id_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(Thal_assy)
        writer.writerows(PN_assy)

    burst_fr = 50.  # Hz. Poisson mean firing rate for burst input
    thal_baseline_fr = 20.0  # Hz. Firing rate for thalamus baseline input
    int_baseline_fr = 20.0  # Hz. Firing rate for interneuron baseline input
    t_start = 1.0  # sec. Time to start burst input
    on_time = 0.5  # sec. Burst input duration
    off_time = 1.0  # sec. Silence duration
    sim_time = (0, t_stop)  # Whole simulation

    firing_rate = (thal_baseline_fr + burst_fr, thal_baseline_fr)

    # Short burst input for 1000 ms followed by 500 ms silence
    psg = get_psg_short(Thal_assy, firing_rate, on_time, off_time,
                        t_start=t_start, t_stop=t_stop)
    psg.to_sonata(os.path.join(INPUT_PATH, "thalamus_short.h5"))

    # Long burst input for 1000 ms followed by 500 ms silence
    psg = get_psg_long(Thal_assy, firing_rate, on_time, off_time,
                       t_start=t_start, t_stop=t_stop)
    psg.to_sonata(os.path.join(INPUT_PATH, "thalamus_long.h5"))

    # Thalamus baseline
    psg = PoissonSpikeGenerator(population='thalamus')
    psg.add(node_ids=df2node_id(Thal_nodes),
            firing_rate=thal_baseline_fr, times=sim_time)
    psg.to_sonata(os.path.join(INPUT_PATH, "thalamus_base.h5"))

    # Interneuron baseline
    psg = PoissonSpikeGenerator(population='Intbase')
    psg.add(node_ids=df2node_id(nodes['Intbase']),
            firing_rate=int_baseline_fr, times=sim_time)
    psg.to_sonata(os.path.join(INPUT_PATH, "Intbase.h5"))

    print("Core cells: %.3f sec" % (time.perf_counter() - start_timer))

    # These inputs are for the baseline firing rates of the cells in the shell.
    if 'shell' in nodes:
        start_timer = time.perf_counter()
        shell_nodes = {p: get_pop(nodes['shell'], p) for p in pop_names}
        PN_ids = df2node_id(shell_nodes['CP']) + df2node_id(shell_nodes['CS'])
        FSI_ids = df2node_id(shell_nodes['FSI'])
        LTS_ids = df2node_id(shell_nodes['LTS'])

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

        PN_ids, r_PN = effective_cell(PN_ids, effective_shell)
        FSI_ids, r_FSI = effective_cell(FSI_ids, effective_shell)
        LTS_ids, r_LTS = effective_cell(LTS_ids, effective_shell)
        print("Proportion of effective cells in shell.")
        print("%.1f%% effective PN." % (100 * r_PN))
        print("%.1f%% effective FSI." % (100 * r_FSI))
        print("%.1f%% effective LTS." % (100 * r_LTS))

        # Generate Poisson spike trains for shell cells
        psg = PoissonSpikeGenerator(population='shell')
        constant_fr = False
        if constant_fr:
            # Constant mean firing rate for all cells
            psg.add(node_ids=PN_ids, firing_rate=0.5, times=sim_time)
            psg.add(node_ids=FSI_ids, firing_rate=5., times=sim_time)
            psg.add(node_ids=LTS_ids, firing_rate=1., times=sim_time)
        else:
            # Lognormal distributed mean firing rate
            fr = []
            fr.append(psg_lognormal_fr(psg, PN_ids,
                                       mean=0.5, stdev=0.2, times=sim_time))
            fr.append(psg_lognormal_fr(psg, FSI_ids,
                                       mean=5., stdev=3., times=sim_time))
            fr.append(psg_lognormal_fr(psg, LTS_ids,
                                       mean=1., stdev=0.1, times=sim_time))
            fr_file = os.path.join(INPUT_PATH, "Lognormal_FR.csv")
            with open(fr_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(fr)
        psg.to_sonata('./input/shell.h5')

        print("Shell cells: %.3f sec" % (time.perf_counter() - start_timer))

    print("Done!")


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        build_input(*sys.argv[1:])
    else:
        build_input()
