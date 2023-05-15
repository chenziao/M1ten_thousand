import numpy as np
from bmtools.cli.plugins.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

import sys
import os
import csv
import time

randseed = 4321
rng = np.random.default_rng(randseed)

T_SIM = 12.  # sec. Simulation time
N_ASSEMBLIES = 8  # number of assemblies
INPUT_PATH = "./input"
ASSEMBLY_ID_PATH = os.path.join(INPUT_PATH, "Assembly_ids.csv")


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


def get_psg_short(Thal_assy, firing_rate=0., on_time=1.0, off_time=0.5,
                  t_start=0., t_stop=T_SIM):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Short burst is delivered to each assembly sequentially within each cycle.
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies
    t_on = on_time / n_assemblies  # on time for each assembly

    psg = PoissonSpikeGenerator(population='thalamus')
    times = np.array([t_start, t_start + t_on])  # window for each assembly
    for i in range(n_cycle):
        for assy in Thal_assy:
            psg.add(node_ids=assy, firing_rate=firing_rate, times=times)
            times += t_on
        times += off_time
    return psg


def get_psg_long(Thal_assy, firing_rate=0., on_time=1.0, off_time=0.5,
                 t_start=0., t_stop=T_SIM):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time. This repeats until the last on-time can complete before t_stop.
    Long burst is delivered to one assembly in each cycle.
    """
    t_cycle, n_cycle = get_stim_cycle(on_time, off_time, t_start, t_stop)
    n_assemblies = len(Thal_assy)  # number of assemblies

    psg = PoissonSpikeGenerator(population='thalamus')
    times = np.array([t_start, t_start + on_time])  # window for each assembly
    for i in range(n_cycle):
        assy = Thal_assy[i % n_assemblies]
        psg.add(node_ids=assy, firing_rate=firing_rate, times=times)
        times += t_cycle
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
    with open(ASSEMBLY_ID_PATH, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(Thal_assy)
        writer.writerows(PN_assy)

    firing_rate = 50.  # Hz. Poisson mean firing rate for burst input
    thal_baseline_fr = 2.0  # Hz. Firing rate for thalamus baseline input
    int_baseline_fr = 2.0  # Hz. Firing rate for interneuron baseline input
    t_start = 0.5  # sec. Time to start burst input
    on_time = 1.0  # sec. Burst input duration
    off_time = 0.5  # sec. Silence duration
    sim_time = (0, t_stop)  # Whole simulation

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

        psg = PoissonSpikeGenerator(population='shell')
        constant_fr = True
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

    print("Done")


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        build_input(*sys.argv[1:])
    else:
        build_input()
