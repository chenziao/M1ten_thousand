import os, sys
from bmtk.simulator import bionet
import numpy as np
import synapses
import warnings
# import corebmtk
from neuron import h

# def run(config_file):

warnings.simplefilter(action='ignore', category=FutureWarning)
synapses.load(randseed=1111)
from bmtk.simulator.bionet.pyfunction_cache import add_weight_function
	
def lognormal_weight(edge_props, source, target):
    m = edge_props.get('syn_weight')
    s = edge_props.get('weight_sigma')
    if m is not None:
        if s:
            weight = synapses.gen_logn_weight(m, s,
                sigma_lower=edge_props.get('sigma_lower_bound'),
                sigma_upper=edge_props.get('sigma_upper_bound'))
        else:
            weight = m
    return weight

add_weight_function(lognormal_weight)
config_file = 'config.json' ######
conf = bionet.Config.from_json(config_file, validate=True)
# conf = corebmtk.Config.from_json(config_file, validate=True)
conf.build_env()

graph = bionet.BioNetwork.from_config(conf)


# This fixes the morphology error in LFP calculation
pop = graph._node_populations['cortex']
for node in pop.get_nodes():
    node._node._node_type_props['morphology'] = node.model_template[1]

sim = bionet.BioSimulator.from_config(conf, network=graph)
# sim = corebmtk.CoreBioSimulator.from_config(conf, network=graph, gpu=False)

# This calls insert_mechs() on each cell to use its gid as a seed
# to the random number generator, so that each cell gets a different
# random seed for the point-conductance noise
cells = graph.get_local_cells()
for cell in cells:
    cells[cell].hobj.insert_mechs(cells[cell].gid)
    pass

sim.run()

bionet.nrn.quit_execution()


# if __name__ == '__main__':
#     if __file__ != sys.argv[-1]:
#         run(sys.argv[-1])
#     else:
#         run('config.json')
