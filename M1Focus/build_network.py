import numpy as np
import os
from functools import partial
from bmtk.builder import NetworkBuilder
from bmtk.utils.sim_setup import build_env_bionet
import synapses
from connectors import (
    spherical_dist, cylindrical_dist_z, GaussianDropoff, UniformInRange,
    pr_2_rho, ReciprocalConnector, UnidirectionConnector,
    OneToOneSequentialConnector, syn_dist_delay_feng_section,
    syn_dist_delay_feng, syn_uniform_delay_section
)

##############################################################################
############################## General Settings ##############################

randseed = 123412
np.random.seed(randseed)

network_dir = 'network'
t_sim = 11500.0
dt = 0.05

# Network size and dimensions
num_cells = 1000  # 10000
column_width, column_height = 1000., 500.
x_start, x_end = -column_width/2, column_width/2
y_start, y_end = -column_width/2, column_width/2
z_start, z_end = 0., column_height
z_5A = 250.

# Distance constraint for all cells
min_conn_dist = 0.0
max_conn_dist = 300.0  # or np.inf
# Distance range for total probability in estimation of Gaussian drop function
ptotal_dist_range = (0., 150.)

# When enabled, a shell of virtual cells will be created around the core cells.
edge_effects = True

##############################################################################
####################### Cell Proportions and Positions #######################

def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))  # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)


# Number of cells in each population.
# Following 80/20 E/I equal on CP-CS and 60% FSI to 40% LTS for Interneurons
# Densities by cell proportion unless otherwise specified:
# CP: 20%  CS: 20% CTH: 20% CC: 20% FSI: 12% LTS: 8%
# Corticopontine, Corticostriatal,
# Fast Spiking Interneuron, Low Threshold Spiker
num_CP, num_CS, num_FSI, num_LTS = num_prop([40, 40, 12, 8], num_cells)
# num_CTH = int(num_cells * 0.2)  # Corticothalamic
# num_CC = int(num_cells * 0.2)   # Corticocortical

# Amount of cells per layer
# CP cells are basically only in layer 5B and nowhere else.
numCP_in5A, numCP_in5B = num_prop([5, 95], num_CP)
# CS cells span top of 5B to middle of 2/3
numCS_in5A, numCS_in5B = num_prop([95, 5], num_CS)
# Even distribution of FSI cells between Layers 5A and 5B
numFSI_in5A, numFSI_in5B = num_prop([1, 1], num_FSI)
# Even distribution of LTS cells between Layers 5A and 5B
numLTS_in5A, numLTS_in5B = num_prop([1, 1], num_LTS)

# Total 400x400x1820 (ignoring layer 1)
# Order from top to bottom is 2/3, 4, 5A, 5B, 6
# Layer 2/3 (420 um thick) 23.1%
# Layer 5A (250 um thick) 13.7% (z is 250 to 499)
# Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
num_cells_5A = numCP_in5A + numCS_in5A + numFSI_in5A + numLTS_in5A
num_cells_5B = numCP_in5B + numCS_in5B + numFSI_in5B + numLTS_in5B

# TODO: Use poisson-disc sampling to generate positions with minimum distance
# limit, possibly using scipy.stats.qmc.PoissonDisk (new in scipy 1.10.0)
pos_list_5A = np.random.rand(num_cells_5A, 3)
pos_list_5A[:, 0] = pos_list_5A[:, 0] * (x_end - x_start) + x_start
pos_list_5A[:, 1] = pos_list_5A[:, 1] * (y_end - y_start) + y_start
pos_list_5A[:, 2] = pos_list_5A[:, 2] * (z_end - z_5A) + z_5A

pos_list_5B = np.random.rand(num_cells_5B,3)
pos_list_5B[:, 0] = pos_list_5B[:, 0] * (x_end - x_start) + x_start
pos_list_5B[:, 1] = pos_list_5B[:, 1] * (y_end - y_start) + y_start
pos_list_5B[:, 2] = pos_list_5B[:, 2] * (z_5A - z_start) + z_start


## TODO: generate random orientations


##############################################################################
####################### Functions for Building Network #######################

def build_networks(network_definitions: list) -> dict:
    """
    `network_definitions` should be a list of dictionaries, e.g. [{}, {}, ...]
    Keys should include an arbitrary `network_name`, a positions_list (if any),
    and `cells`. `cells` should contain a list of dictionaries, and each
    dictionary should corrospond with any valid input for BMTK
    NetworkBuilder.add_nodes() method. A dictionary of BMTK NetworkBuilder
    objects will be returned, reference by individual network_name."""
    for net_def in network_definitions:
        network_name = net_def['network_name']
        if networks.get(network_name) is None:
            networks[network_name] = NetworkBuilder(network_name)
        pos_list = net_def.get('positions_list')

        # Add cells to the network
        num = 0
        for cell in net_def['cells']:
            num_cells = cell['N']
            extra_kwargs = {}
            if pos_list is not None:
                extra_kwargs['positions'] = pos_list[num:num + num_cells]
                num += num_cells

            cell = {k: v for k, v in cell.items() if v is not None}
            extra_kwargs = {k: v for k, v in extra_kwargs.items()
                            if v is not None}
            networks[network_name].add_nodes(**cell, **extra_kwargs)

    return networks


def build_edges(networks, edge_definitions, edge_params,
                edge_add_properties, syn):
    """
    Builds the edges for each network given a set of 'edge_definitions',
    examples shown later in the code
    """
    for edge in edge_definitions:
        network_name = edge['network']
        net = networks[network_name]
        # edge arguments
        print("Adding edge: " + edge['param'])
        edge_params_val = edge_params[edge['param']].copy()
        # get synapse template file
        dynamics_file = edge_params_val.pop('dynamics_params')
        model_template = syn[dynamics_file]['level_of_detail']
        # get source and target nodes
        edge_src_trg = edge.get('edge')
        if edge_src_trg:
            edge_src_trg = edge_src_trg.copy()
            src_net = edge_src_trg.pop('source_network', network_name)
            trg_net = edge_src_trg.pop('target_network', network_name)
            source = networks[src_net].nodes(**edge_src_trg.get('source', {}))
            target = networks[trg_net].nodes(**edge_src_trg.get('target', {}))
            edge_params_val.update({'source': source, 'target': target})
        # use connector class
        connector_class = edge_params_val.pop('connector_class', None)
        if connector_class is not None:
            # create a connector object
            connector_params = edge_params_val.pop('connector_params', {})
            connector = connector_class(**connector_params)
            # keep object reference in the dictionary
            edge_params[edge['param']]['connector_object'] = connector
            if edge_src_trg:
                connector.setup_nodes(source=source, target=target)
            edge_params_val.update(connector.edge_params())
        conn = net.add_edges(model_template=model_template, **edge_params_val)

        edge_properties = edge.get('add_properties')
        if edge_properties:
            edge_properties_val = edge_add_properties[edge_properties].copy()
            if connector_class is not None:
                # pass connector object to the rule for edge properties
                edge_properties_val['rule'] = partial(
                    edge_properties_val['rule'], connector=connector)
            conn.add_properties(**edge_properties_val)


def get_connector(param):
    """Get connector object stored in edge_params"""
    edge_params_val = edge_params[param]
    if 'connector_object' in edge_params_val:
        return edge_params_val['connector_object']
    else:
        raise ValueError("No connector used in '%s'" % param)


def save_networks(networks, network_dir):
    """Build and save network"""
    # Remove the existing network_dir directory
    for f in os.listdir(network_dir):
        os.remove(os.path.join(network_dir, f))

    # Run through each network and save their nodes/edges
    for network_name, network in networks.items():
        print('Building ' + network_name)
        network.build()
        network.save_nodes(output_dir=network_dir)
        network.save_edges(output_dir=network_dir)


##############################################################################
############################ Network Definitions #############################

# Dictionary to store NetworkBuilder objects referenced by name
networks = {}
network_definitions = [
    {   # Start Layer 5A
        'network_name': 'cortex',
        'positions_list': pos_list_5A,
        'cells': [
            {   # CP
                'N': numCP_in5A,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_in5A,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_in5A,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_in5A,
                'pop_name': 'LTS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    },  # End Layer 5A
    {   # Start Layer 5B
        'network_name': 'cortex',
        'positions_list': pos_list_5B,
        'cells': [
            {   # CP
                'N': numCP_in5B,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_in5B,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_in5B,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_in5B,
                'pop_name': 'LTS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    },  # End Layer 5B
    {   # Extrinsic Thalamic Inputs
        'network_name': 'thalamus',
        'positions_list': None,
        'cells': [
            {   # Virtual Cells
                'N': num_CP + num_CS,
                'pop_name': 'thal',
                'potential': 'exc',
                'model_type': 'virtual'
            }
        ]
    },
    {   # Extrinsic Intbase Inputs
        'network_name': 'Intbase',
        'positions_list': None,
        'cells': [
            {   # Virtual Cells
                'N': num_FSI + num_LTS,
                'pop_name': 'Int',
                'potential': 'exc',
                'model_type': 'virtual'
            }
        ]
    }
]


##############################################################################
################################ EDGE EFFECTS ################################

if edge_effects:
    # Compute the outer shell range. The absolute max_conn_dist will extend
    # each dimension of the core by 2 * max_conn_dist
    shell_x_start, shell_y_start, shell_z_start = \
        np.array((x_start, y_start, z_start)) - max_conn_dist
    shell_x_end, shell_y_end, shell_z_end = \
        np.array((x_end, y_end, z_end)) + max_conn_dist

    # Compute the core and shell volume
    core_volume_5A = (x_end - x_start) * (y_end - y_start) * (z_end - z_5A)
    core_volume_5B = (x_end - x_start) * (y_end - y_start) * (z_5A - z_start)
    shell_volume_5A = (shell_x_end - shell_x_start) * \
        (shell_y_end - shell_y_start) * (shell_z_end - z_5A)
    shell_volume_5B = (shell_x_end - shell_x_start) * \
        (shell_y_end - shell_y_start) * (z_5A - shell_z_start)

    # Increase the number of original cells based on the volume difference
    # between core and shell
    # Layer 5A
    virt_num_cells_5A = int(round(num_cells_5A *
                                  shell_volume_5A / core_volume_5A))
    # Layer 5B
    virt_num_cells_5B = int(round(num_cells_5B *
                                  shell_volume_5B / core_volume_5B))

    # Create a positions list for cells in the shell
    virt_pos_list_5A = np.random.rand(virt_num_cells_5A, 3)
    virt_pos_list_5A[:, 0] = virt_pos_list_5A[:, 0] * \
        (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5A[:, 1] = virt_pos_list_5A[:, 1] * \
        (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5A[:, 2] = virt_pos_list_5A[:, 2] * \
        (shell_z_end - z_5A) + z_5A
    i_shell = (virt_pos_list_5A[:, 2] > z_end) | \
        (virt_pos_list_5A[:, 0] < x_start) | (virt_pos_list_5A[:, 0] > x_end) | \
        (virt_pos_list_5A[:, 1] < y_start) | (virt_pos_list_5A[:, 1] > y_end)

    virt_pos_list_5A = virt_pos_list_5A[i_shell]

    virt_pos_list_5B = np.random.rand(virt_num_cells_5B, 3)
    virt_pos_list_5B[:, 0] = virt_pos_list_5B[:, 0] * \
        (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5B[:, 1] = virt_pos_list_5B[:, 1] * \
        (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5B[:, 2] = virt_pos_list_5B[:, 2] * \
        (z_5A - shell_z_start) + shell_z_start
    i_shell = (virt_pos_list_5B[:, 2] < z_start) | \
        (virt_pos_list_5B[:, 0] < x_start) | (virt_pos_list_5B[:, 0] > x_end) | \
        (virt_pos_list_5B[:, 1] < y_start) | (virt_pos_list_5B[:, 1] > y_end)

    virt_pos_list_5B = virt_pos_list_5B[i_shell]

    # Recalculate number of cells in each layer
    virt_num_cells_5A = len(virt_pos_list_5A)
    virt_numCP_in5A, virt_numCS_in5A, virt_numFSI_in5A, virt_numLTS_in5A = \
        num_prop([numCP_in5A, numCS_in5A, numFSI_in5A, numLTS_in5A],
                 virt_num_cells_5A)

    virt_num_cells_5B = len(virt_pos_list_5B)
    virt_numCP_in5B, virt_numCS_in5B, virt_numFSI_in5B, virt_numLTS_in5B = \
        num_prop([numCP_in5B, numCS_in5B, numFSI_in5B, numLTS_in5B],
                 virt_num_cells_5B)

    virt_num_cells = virt_num_cells_5A + virt_num_cells_5B

    # This network should contain all the same properties as the original
    # network, except the cell should be virtual. For connectivity, you should
    # name the cells the same as the original network because connection rules
    # defined later will require it
    shell_network = [
    {   # Start Layer 5A
        'network_name': 'shell',
        'positions_list': virt_pos_list_5A,
        'cells': [
            {   # CP
                'N': virt_numCP_in5A,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_in5A,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_in5A,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_in5A,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    },  # End Layer 5A
    {   # Start Layer 5B
        'network_name': 'shell',
        'positions_list': virt_pos_list_5B,
        'cells': [
            {   # CP
                'N': virt_numCP_in5B,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_in5B,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_in5B,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_in5B,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    }  # End Layer 5B
    ]

    # Add the shell to our network definitions
    network_definitions.extend(shell_network)

############################## END EDGE EFFECTS ##############################
##############################################################################

# Build and save our NetworkBuilder dictionary
networks = build_networks(network_definitions)


##########################################################################
#############################  BUILD EDGES  ##############################

# Whole reason for restructuring network building lies here, by separating out
# the source and target params from the remaining parameters in
# NetworkBuilder.add_edges() function we can reuse connectivity rules for the
# virtual shell or elsewhere
# [
#  {
#   'network': 'network_name',  # Name of the network to which edges are added
#   'edge': {
#       'source': {},  # dictionary of properties of desired population
#       'target': {},
#       'source_network': 'network_name1'  # network name for the population
#       'target_network': 'network_name2'  # if different from network_name
#       },  # source and target are required
#       # source_network and target_network are optional
#   'param': 'name_of_edge_parameters',  # to be coupled with when add_edges()
#   'add_properties': 'prop_name'  # name of edge_add_properties for additional
#       # connection properties, like delay
#   }
# ]


edge_definitions = [
    {   # CP -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # FSI -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP -> FSI forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP <- FSI backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS -> FSI forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS <- FSI backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # LTS -> LTS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # PN -> LTS forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP', 'CS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'PN2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # PN <- LTS backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP', 'CS']}
        },
        'param': 'LTS2PN',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # FSI -> LTS forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # FSI <- LTS backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
        ################### THALAMIC INPUT ###################
    {   # Thalamus Excitation to CP
        'network': 'cortex',
        'edge': {
            'source_network': 'thalamus',
            'source': {},
            'target': {'pop_name': ['CP']}
        },
        'param': 'Thal2CP'
    },
    {   # Thalamus Excitation to CS
        'network': 'cortex',
        'edge': {
            'source_network': 'thalamus',
            'source': {},
            'target': {'pop_name': ['CS']}
        },
        'param': 'Thal2CS'
    },
        ################### Interneuron baseline INPUT ###################
    {   # Excitation to FSI
        'network': 'cortex',
        'edge': {
            'source_network': 'Intbase',
            'source': {},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'Intbase2FSI'
    },
    {   # Excitation to LTS
        'network': 'cortex',
        'edge': {
            'source_network': 'Intbase',
            'source': {},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'Intbase2LTS'
    }
]

# edge_params should contain additional parameters to be added to add_edges()
# The following parameters for random synapse placement are not necessary
# in edge_params if sec_id and sec_x are specified instead.
# distance_range: place synapse within distance range [dmin, dmax] from soma
# target_sections: place synapse within the given sections in a list
edge_params = {
    'CP2CP': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=124.62, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.1 * (0.5 + 0.3 / 0.7),
                ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),  # 10% unidirectional
            'p0_arg': cylindrical_dist_z,
            'pr': 0.1 / 0.7 * 0.3,  # 30% of connected paris are reciprocal
            'estimate_rho': True
            },
        'syn_weight': 1,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'CP2CP.json'
    },
    'CS2CS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=124.62, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.1 * (0.5 + 0.3 / 0.7),
                ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),
            'p0_arg': cylindrical_dist_z,  # 10% unidirectional
            'pr': 0.1 / 0.7 * 0.3,  # 30% of connected paris are reciprocal
            'estimate_rho': True
            },
        'syn_weight': 1,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'CS2CS.json'
    },
    'CP2CS': {
        'connector_class': UnidirectionConnector,
        'connector_params': {
            'p': GaussianDropoff(
                stdev=124.62, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.01, ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),
            'p_arg': cylindrical_dist_z,
            },
        'syn_weight': 1,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'CP2CS.json'  # same as 'CS2CS.json'
    },
    'CS2CP': {
        'connector_class': UnidirectionConnector,
        'connector_params': {
            'p': GaussianDropoff(
                stdev=124.62, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.1, ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),
            'p_arg': cylindrical_dist_z,
            },
        'syn_weight': 1,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'CS2CS.json'
    },
    'FSI2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=131.48, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.34 / 2 + 0.43, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 34% unidirectional
            'p0_arg': spherical_dist,
            'pr': 0.43, 'estimate_rho': True  # 43% reciprocal
            # 'rho': pr_2_rho(0.6, 0.6, 0.43)  # use fixed rho instead
            },
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'FSI2FSI.json'
    },
    'CP2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=99.25, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.32, ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),  # 32% total
            'p0_arg': cylindrical_dist_z,
            'p1': GaussianDropoff(
                stdev=95.98, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.20 + 0.28, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 20% unidirectional
            'p1_arg': spherical_dist,
            'pr': 0.28, 'estimate_rho': True  # 28% reciprocal
            },
        'syn_weight': 1,
        'sec_id': 1,  # dend
        'sec_x': 0.5,
        'dynamics_params': 'CP2FSI.json'
    },
    'FSI2CP': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CP2FSI'},
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'FSI2CP.json'
    },
    'CS2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=99.25, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.22, ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),  # 22% total
            'p0_arg': cylindrical_dist_z,
            'p1': GaussianDropoff(
                stdev=95.98, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.17 + 0.20, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 17% unidirectional
            'p1_arg': spherical_dist,
            'pr': 0.20, 'estimate_rho': True  # 20% reciprocal
            },
        'syn_weight': 1,
        'sec_id': 1,  # dend
        'sec_x': 0.5,
        'dynamics_params': 'CS2FSI.json'
    },
    'FSI2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CS2FSI'},
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'FSI2CS.json'
    },
    'LTS2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=131.48, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.034 / 2 + 0.043, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 3.4% unidirectional
            'p0_arg': spherical_dist,
            'pr': 0.043, 'estimate_rho': True  # 4.3% reciprocal
            },
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'INT2INT.json'
    },
    'PN2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=99.25, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.57, ptotal_dist_range=ptotal_dist_range,
                dist_type='cylindrical'),
            'p0_arg': cylindrical_dist_z,
            'p1': GaussianDropoff(
                stdev=95.98, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.35, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),
            'p1_arg': spherical_dist,
            'pr': 0.2, 'estimate_rho': True
            },
        'syn_weight': 1,
        'sec_id': 1,  # dend
        'sec_x': 0.5,
        'dynamics_params': 'PN2LTS.json'
    },
    'LTS2PN': {
        'connector_class': get_connector,
        'connector_params': {'param': 'PN2LTS'},
        'syn_weight': 1,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'LTS2PN.json'
    },
    'FSI2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=131.48, min_dist=min_conn_dist, max_dist=max_conn_dist,
                ptotal=0.34 + 0.21, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 34% unidirectional
            'p0_arg': spherical_dist,
            'p1': GaussianDropoff(
                stdev=95.98, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.53 + 0.21, ptotal_dist_range=ptotal_dist_range,
                dist_type='spherical'),  # 53% unidirectional
            'p1_arg': spherical_dist,
            'pr': 0.21, 'estimate_rho': True  # 21% reciprocal
            },
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'FSI2LTS.json'
    },
    'LTS2FSI': {
        'connector_class': get_connector,
        'connector_params': {'param': 'FSI2LTS'},
        'syn_weight': 1,
        'sec_id': 0,  # soma
        'sec_x': 0.5,
        'dynamics_params': 'LTS2FSI.json'
    },
    'Thal2CP': {
        'connector_class': OneToOneSequentialConnector,
        'syn_weight': 1,
        'delay': 0.0,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'Thal2CP.json'
    },
    'Thal2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Thal2CP'},
        'syn_weight': 1,
        'delay': 0.0,
        'sec_id': 2,
        'sec_x': 0.9,  # end of apic
        'dynamics_params': 'Thal2CS.json'  # same as 'Thal2CP.json'
    },
    'Intbase2FSI': {
        'connector_class': OneToOneSequentialConnector,
        'syn_weight': 1,
        'delay': 0.0,
        'sec_id': 1,  # dend
        'sec_x': 0.5,
        'dynamics_params': 'Intbase2FSI.json'
    },
    'Intbase2LTS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Intbase2FSI'},
        'syn_weight': 1,
        'delay': 0.0,
        'sec_id': 1,  # dend
        'sec_x': 0.5,
        'dynamics_params': 'Intbase2LTS.json'
    }
}  # edges referenced by name

# Will be called by conn.add_properties() for the associated connection
edge_add_properties = {
    'syn_dist_delay_feng_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_dist_delay_feng_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [float, np.int32, float]
    },
    'syn_dist_delay_feng_default': {
        'names': 'delay',
        'rule': syn_dist_delay_feng,
        'dtypes': float
    },
    'syn_uniform_delay_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_uniform_delay_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [float, np.int32, float]
    }
}


# Load synapse dictionaries
# See synapses.py - loads each json's in components/synaptic_models into a
# dictionary so the properties can be referenced in the files,
# e.g., syn['file.json'].get('property')
synapses.load()
syn = synapses.syn_params_dicts()

# Build your edges into the networks
build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn)


##############################################################################
############################  EDGE EFFECTS EDGES  ############################

if edge_effects:
    # These rules are for edge effect edges. They should mimic the connections
    # created previously but using unidirectional connector.
    # Re-use the connector params set above.

    # Find core network edge types that need shell connections
    core_network_name = 'cortex'
    core_edge_def = []
    for edge in edge_definitions:
        network_name = edge['network']
        if network_name != core_network_name:
            continue
        is_core = True
        edge_src_trg = edge.get('edge')
        if edge_src_trg:
            for net_type in ('source_network', 'target_network'):
                net_name = edge_src_trg.get(net_type)
                if net_name is not None and net_name != core_network_name:
                    is_core = False
        if is_core:
            core_edge_def.append(edge)

    # Automatically set up network edges and parameters for shell network
    # Only connections from shell to core is needed, so UnidirectionConnector
    # is used, and parameters are extracted from connectors used in core edges.
    shell_network_name = 'shell'
    shell_edges = []
    shell_edge_params = {}
    for edge in core_edge_def:
        shell_edge = edge.copy()
        edge_src_trg = shell_edge.get('edge')
        if edge_src_trg:
            edge_src_trg['source_network'] = shell_network_name
        shell_edge['param'] = shell_network_name + shell_edge['param']
        shell_edges.append(shell_edge)

        edge_params_val = edge_params[edge['param']].copy()
        connector = edge_params_val.pop('connector_object', None)
        connector_class = edge_params_val.get('connector_class')
        if (connector_class is not None and
                connector_class is not UnidirectionConnector):
            replace = True
            var_list = ('p', 'p_arg', 'n_syn')
            if connector_class is ReciprocalConnector:
                var_map = ('p0', 'p0_arg', 'n_syn0')
            elif connector_class is get_connector:
                var_map = ('p1', 'p1_arg', 'n_syn1')
            else:
                replace = False
                print("Warning: Connector method not identified. "
                      "Use the same connector class for shell edges.")
            if replace:
                edge_params_val['connector_class'] = UnidirectionConnector
                connector_params = {
                    k: connector.vars[k0] for k, k0 in zip(var_list, var_map)
                    }
                connector_params['verbose'] = connector.verbose
                edge_params_val['connector_params'] = connector_params
        shell_edge_params[shell_edge['param']] = edge_params_val

    # Check parameters
    print("\nShell edges:")
    for shell_edge in shell_edges:
        print(shell_edge)
    print("\nShell edge parameters:")
    for param, edge_params_val in shell_edge_params.items():
        print(param + ':')
        print(edge_params_val)
    print("")

    # Build your shell edges into the networks
    build_edges(networks, shell_edges, shell_edge_params,
                edge_add_properties, syn)

########################## END EDGE EFFECTS ##############################
##########################################################################


##########################################################################
############################ GAP JUNCTIONS ###############################
# FSI
net = networks['cortex']
population = net.nodes(pop_name='FSI')

gap_junc_FSI = UnidirectionConnector(
    p=UniformInRange(
        p=0.4, min_dist=min_conn_dist, max_dist=max_conn_dist
        ),
    p_arg=spherical_dist, n_syn=1
)
gap_junc_FSI.setup_nodes(source=population, target=population)

conn = net.add_gap_junctions(resistance=1500, target_sections=None,
                             **gap_junc_FSI.edge_params())
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.5

# LTS
net = networks['cortex']
population = net.nodes(pop_name='LTS')

gap_junc_LTS = UnidirectionConnector(
    p=UniformInRange(
        p=0.3, min_dist=min_conn_dist, max_dist=max_conn_dist
        ),
    p_arg=spherical_dist, n_syn=1
)
gap_junc_LTS.setup_nodes(source=population, target=population)

conn = net.add_gap_junctions(resistance=1500, target_sections=None,
                             **gap_junc_LTS.edge_params())
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.5

##########################################################################
###############################  BUILD  ##################################


# Save the network into the appropriate network dir
save_networks(networks, network_dir)

# Usually not necessary if you've already built your simulation config
if False:
    build_env_bionet(
        base_dir='./',
        network_dir=network_dir,
        tstop=t_sim,
        dt=dt,
        report_vars=['v'],
        celsius=31.0,
        spikes_inputs=[
            ('thalamus', './input/thalamus_base.h5'),
            ('thalamus', './input/thalamus_short.h5'),
            ('thalamus', './input/thalamus_long.h5'),
            ('Intbase', './input/Intbase.h5')
        ],  # (Population for which spikes will be generated, file name)
        components_dir='components',
        config_file='config.json',
        compile_mechanisms=False
    )
