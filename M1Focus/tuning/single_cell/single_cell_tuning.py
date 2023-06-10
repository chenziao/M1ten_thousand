from bmtool.singlecell import Profiler
from multiprocessing import Process
from neuron import h

template_dir = '../../components/templates'
mechanism_dir = '../../components/mechanisms'

def passive_properties(Cell, **kwargs):
    profiler = Profiler(template_dir=template_dir, mechanism_dir=mechanism_dir)
    h.steps_per_ms = 1 / h.dt

    time_vec, voltage_vec = profiler.passive_properties(Cell, **kwargs)
    return time_vec, voltage_vec

def current_injection(Cell, noise=False, **kwargs):
    profiler = Profiler(template_dir=template_dir, mechanism_dir=mechanism_dir)
    h.steps_per_ms = 1 / h.dt

    post_init_function = 'insert_mechs(0)' if noise else None
    time_vec, voltage_vec = profiler.current_injection(
        Cell, post_init_function=post_init_function, **kwargs
    )
    return time_vec, voltage_vec

def fi_curve(Cell, noise=False, **kwargs):
    profiler = Profiler(template_dir=template_dir, mechanism_dir=mechanism_dir)
    h.steps_per_ms = 1 / h.dt

    post_init_function = 'insert_mechs(0)' if noise else None
    amp_vec, spike_vec = profiler.fi_curve(
        Cell, post_init_function=post_init_function, **kwargs
    )
    return amp_vec, spike_vec


Cell = 'CP_Cell'
h.celsius = 26.
h.dt = 0.05

if __name__ ==  '__main__':
    p1 = Process(
        target=passive_properties,
        kwargs={
            'Cell': Cell, 'inj_amp':-20., 'inj_delay': 400., 'inj_dur': 1000.,
            'tstop': 1600., 'method': 'exp2'
        }
    )
    p1.start()
    p1.join()
    
    p2 = Process(
        target=current_injection,
        kwargs={
            'Cell': Cell, 'noise': False,
            'inj_amp': 200., 'inj_delay': 400.,
            'inj_dur': 1000., 'tstop': 1700.
        }
    )
    p2.start()
    p2.join()
    
    p3 = Process(
        target=fi_curve,
        kwargs={
            'Cell': Cell, 'noise': False,
            'i_start': 0., 'i_stop': 1000.,
            'i_increment': 50., 'tstart': 400.
        }
    )
    p3.start()
    p3.join()
