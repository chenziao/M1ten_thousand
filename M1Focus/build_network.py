import numpy as np
import os
from functools import partial
from bmtk.builder import NetworkBuilder
from bmtk.builder.node_pool import NodePool
from bmtk.utils.sim_setup import build_env_bionet
import synapses
from connectors import (
    ReciprocalConnector, GaussianDropoff, spherical_dist, cylindrical_dist_z,
    syn_dist_delay_feng_section, syn_uniform_delay_section, section_id_placement
)

randseed = 123412
np.random.seed(randseed)

network_dir = 'network'
t_sim = 11500.0
dt = 0.05

num_cells = 1000 # 10000
column_width, column_height = 1000., 1000.
x_start, x_end = -column_width/2, column_width/2
y_start, y_end = -column_width/2, column_width/2
z_start, z_end = 0., column_height
z_5A = 500.

min_conn_dist = 0.0 # Distance constraint for all cells
max_conn_dist = 300.0 # 300.0 #9999.

# When enabled, a shell of virtual cells will be created around the core network.
edge_effects = True

###################################################################################
####################### Cell Proportions and Positions ############################
def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0)) # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)

# Number of cells in each population. Following 80/20 E/I equal on CP-CS and 60% FSI to 40% LTS for Interneurons
# Densities by cell proportion unless otherwise specified: CP: 20%  CS: 20% CTH: 20% CC: 20% FSI: 12% LTS: 8%
# Corticopontine, Corticostriatal, Fast Spiking Interneuron, Low Threshold Spiker
num_CP, num_CS, num_FSI, num_LTS = num_prop([40, 40, 12, 8], num_cells)
# num_CTH = int(num_cells * 0.2)  # Corticothalamic
# num_CC = int(num_cells * 0.2)   # Corticocortical

# amount of cells per layer
numCP_in5A, numCP_in5B = num_prop([5, 95], num_CP) # CP cells are basically only in layer 5B and nowhere else.
numCS_in5A, numCS_in5B = num_prop([95, 5], num_CS) # CS cells span top of 5B to middle of 2/3

numFSI_in5A, numFSI_in5B = num_prop([1, 1], num_FSI) # Even distribution of FSI cells between Layers 5A and 5B
numLTS_in5A, numLTS_in5B = num_prop([1, 1], num_LTS) # Even distribution of LTS cells between Layers 5A and 5B

# total 400x400x1820 (ignoring layer 1)
# Order from top to bottom is 2/3, 4, 5A, 5B, 6
# Layer 2/3 (420 um thick) 23.1%
# Layer 5A (250 um thick) 13.7% (z is 250 to 499)
# Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
num_cells_5A = numCP_in5A + numCS_in5A + numFSI_in5A + numLTS_in5A
num_cells_5B = numCP_in5B + numCS_in5B + numFSI_in5B + numLTS_in5B

pos_list_5A = np.random.rand(num_cells_5A, 3)
pos_list_5A[:,0] = pos_list_5A[:,0] * (x_end - x_start) + x_start
pos_list_5A[:,1] = pos_list_5A[:,1] * (y_end - y_start) + y_start
pos_list_5A[:,2] = pos_list_5A[:,2] * (z_end - z_5A) + z_5A

pos_list_5B = np.random.rand(num_cells_5B,3)
pos_list_5B[:,0] = pos_list_5B[:,0] * (x_end - x_start) + x_start
pos_list_5B[:,1] = pos_list_5B[:,1] * (y_end - y_start) + y_start
pos_list_5B[:,2] = pos_list_5B[:,2] * (z_5A - z_start) + z_start

## TODO: generate random orientations


def build_networks(network_definitions: list) -> dict:
    # network_definitions should be a list of dictionaries, e.g. [{}]
    # Keys should include an arbitrary 'network_name', a positions_list (if any),
    # And 'cells'. 'cells' should contain a list of dictionaries, and the dictionary
    # should corrospond with any valid input for BMTK's NetworkBuilder.add_nodes method
    # A dictionary of NetworkBuilder BMTK objects will be returned, reference by individual network_name
    for net_def in network_definitions:
        network_name = net_def['network_name']
        if networks.get(network_name) is None:
            networks[network_name] = NetworkBuilder(network_name)  # This is changed
        pos_list = net_def.get('positions_list', None)

        # Add cells to the network
        num = 0
        for cell in net_def['cells']:
            num_cells = cell['N']
            extra_kwargs = {}
            if pos_list is not None:
                extra_kwargs['positions'] = pos_list[num:num + num_cells]
                num += num_cells

            cell = {k: v for k, v in cell.items() if v is not None}
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            networks[network_name].add_nodes(**cell, **extra_kwargs)

    return networks

def build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn=None):
    # Builds the edges for each network given a set of 'edge_definitions'
    # edge_definitions examples shown later in the code
    for edge in edge_definitions:
        net = networks[edge['network']]
        # edge arguments
        edge_params_val = edge_params[edge['param']].copy()
        # get synapse template file
        dynamics_file = edge_params_val.pop('dynamics_params')
        model_template = syn[dynamics_file]['level_of_detail']
        # get source and target nodes
        edge_src_trg = edge.get('edge')
        if edge_src_trg:
            src_net = edge_src_trg.get('source_network', net)
            trg_net = edge_src_trg.get('target_network', net)
            source = src_net.nodes(**edge_src_trg['source'])
            target = trg_net.nodes(**edge_src_trg['target'])
            edge_params_val.update({'source': source, 'target': target})
        # use connector class
        connector_class = edge_params_val.pop('connector_class', None)
        if connector_class is not None:
            # create a connector object
            connector_params = edge_params_val.pop('connector_params')
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
                edge_properties_val['rule'] = partial(
                    edge_properties_val['rule'], connector=connector)
            conn.add_properties(**edge_properties_val)

def save_networks(networks,network_dir):
    # Remove the existing network_dir directory
    for f in os.listdir(network_dir):
        os.remove(os.path.join(network_dir, f))

    # Run through each network and save their nodes/edges
    for network_name, network in networks.items():
        print('Building ' + network_name)
        network.build()
        network.save_nodes(output_dir=network_dir)
        network.save_edges(output_dir=network_dir)


networks = {}   # Dictionary to store NetworkBuilder objects referenced by name
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

##########################################################################
############################  EDGE EFFECTS  ##############################

if edge_effects: # When enabled, a shell of virtual cells will be created around the core network.

    # compute the outer shell range. The absolute max_conn_dist will extend each dimension of the core by 2*max_conn_dist
    shell_x_start, shell_y_start, shell_z_start = np.array((x_start, y_start, z_start)) - max_conn_dist
    shell_x_end, shell_y_end, shell_z_end = np.array((x_end, y_end, z_end)) + max_conn_dist

    # compute the core and shell volume
    core_volume_5A = (x_end - x_start) * (y_end - y_start) * (z_end - z_5A)
    core_volume_5B = (x_end - x_start) * (y_end - y_start) * (z_5A - z_start)
    shell_volume_5A = (shell_x_end - shell_x_start) * (shell_y_end - shell_y_start) * (shell_z_end - z_5A)
    shell_volume_5B = (shell_x_end - shell_x_start) * (shell_y_end - shell_y_start) * (z_5A - shell_z_start)

    # Increase the number of original cells based on the volume difference between core and shell
    #Layer 5A
    virt_num_cells_5A = int(round(num_cells_5A * shell_volume_5A / core_volume_5A))
    #Layer 5B
    virt_num_cells_5B = int(round(num_cells_5B * shell_volume_5B / core_volume_5B))

    # Create a positions list for cells in the shell
    virt_pos_list_5A = np.random.rand(virt_num_cells_5A, 3)
    virt_pos_list_5A[:,0] = virt_pos_list_5A[:,0] * (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5A[:,1] = virt_pos_list_5A[:,1] * (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5A[:,2] = virt_pos_list_5A[:,2] * (shell_z_end - z_5A) + z_5A
    i_shell = (virt_pos_list_5A[:,0] < x_start) | (virt_pos_list_5A[:,0] > x_end) | \
              (virt_pos_list_5A[:,1] < y_start) | (virt_pos_list_5A[:,1] > y_end) | \
              (virt_pos_list_5A[:,2] > z_end)
    virt_pos_list_5A = virt_pos_list_5A[i_shell]

    virt_pos_list_5B = np.random.rand(virt_num_cells_5B, 3)
    virt_pos_list_5B[:,0] = virt_pos_list_5B[:,0] * (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5B[:,1] = virt_pos_list_5B[:,1] * (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5B[:,2] = virt_pos_list_5B[:,2] * (z_5A - shell_z_start) + shell_z_start
    i_shell = (virt_pos_list_5B[:,0] < x_start) | (virt_pos_list_5B[:,0] > x_end) | \
              (virt_pos_list_5B[:,1] < y_start) | (virt_pos_list_5B[:,1] > y_end) | \
              (virt_pos_list_5B[:,2] < z_start)
    virt_pos_list_5B = virt_pos_list_5B[i_shell]

    # Recalculate number of cells in each layer
    virt_num_cells_5A = len(virt_pos_list_5A)
    virt_numCP_in5A, virt_numCS_in5A, virt_numFSI_in5A, virt_numLTS_in5A = \
        num_prop([numCP_in5A, numCS_in5A, numFSI_in5A, numLTS_in5A], virt_num_cells_5A)

    virt_num_cells_5B = len(virt_pos_list_5B)
    virt_numCP_in5B, virt_numCS_in5B, virt_numFSI_in5B, virt_numLTS_in5B = \
        num_prop([numCP_in5B, numCS_in5B, numFSI_in5B, numLTS_in5B], virt_num_cells_5B)

    virt_num_cells = virt_num_cells_5A + virt_num_cells_5B

    # This network should contain all the same properties as the original network, except
    # the cell should be virtual. For connectivity, you should name the cells the same as
    # the original network because connection rules defined later will require it
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
    }, # End Layer 5A
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
    } # End Layer 5B
]
# Add the shell to our network definitions
network_definitions.extend(shell_network)
########################## END EDGE EFFECTS ##############################
##########################################################################

# Build and save our NetworkBuilder dictionary
networks = build_networks(network_definitions)
# import pdb; pdb.set_trace()

##########################################################################
#############################  BUILD EDGES  ##############################

# Whole reason for restructuring network building lies here, by separating out the
# source and target params from the remaining parameters in NetworkBuilder's
# add_edges function we can reuse connectivity rules for the virtual shell
# or elsewhere
# [
#    {
#       'network': 'network_name', # => The name of the network that these edges should be added to (networks['network_name'])
#       'edge': {
#                    'source': {},
#                    'target': {}
#               }, # should contain source and target only, any valid add_edges param works
#       'param': 'name_of_edge_parameter' # to be coupled with when add_edges is called
#       'add_properties': 'prop_name' # name of edge_add_properties for adding additional connection props, like delay
#    }
# ]

# Will be called by conn.add_properties for the associated connection
edge_add_properties = {
    'syn_dist_delay_feng_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_dist_delay_feng_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [float, np.int32, float]
    },
    'syn_uniform_delay_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_uniform_delay_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [float, np.int32, float]
    },
    'section_id_placement': {
        'names': ['sec_id', 'sec_x'],
        'rule': section_id_placement,
        'dtypes': [np.int32, float]
    }
}

edge_definitions = [
    {   # FSI -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> FSI forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP <- FSI backward
        'network': 'cortex',
        'param': 'FSI2CP',
        'add_properties': 'syn_dist_delay_feng_section_default'
    }

        ################### THALAMIC INPUT ################

        ##################### Interneuron baseline INPUT #####################

]

# A few connectors require a list for tracking synapses that are recurrent, declare them here
FSI_FSI_list = []
FSI_CP_list = []
FSI_CS_list = []
CP_CP_list = []
CS_CS_list = []
LTS_LTS_list = []
LTS_CP_list = []
CP_LTS_list = []
CS_LTS_list = []
FSI_LTS_list = []
LTS_FSI_list = []
# CS_CP_list = []
# CP_CS_list = []

def GetConnector(param):
    edge_params_val = edge_params[param]
    if 'connector_object' in edge_params_val:
        return edge_params_val['connector_object']
    else:
        raise ValueError("No connector implemented in '%s'" % param)

# edge_params should contain additional parameters to be added to add_edges calls
# The following parameters for random synapse placement are not necessary
# if conn.add_properties specifies sec_id and sec_x.
# distance_range: place synapse within distance range [dmin, dmax] from soma
# target_sections: place synapse within the given sections in a list
edge_params = {
    'FSI2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=131.48, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.34, dist_type='spherical'),
            'p0_arg': spherical_dist,
            'pr': 0.34 * 0.43, 'estimate_rho': True,
            },
        'syn_weight': 1,
        'dynamics_params': 'FSI2FSI.json'
    },
    'CP2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': GaussianDropoff(
                stdev=99.25, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.32, dist_type='cylindrical'),
            'p0_arg': cylindrical_dist_z,
            'p1': GaussianDropoff(
                stdev=95.98, min_dist=min_conn_dist, max_dist=max_conn_dist,
                pmax=0.20, dist_type='spherical'),
            'p1_arg': spherical_dist,
            'pr': 0.20 * 0.28, 'estimate_rho': True,
            },
        'syn_weight': 1,
        'dynamics_params': 'CP2FSI.json'
    },
    'FSI2CP': {
        'connector_class': GetConnector,
        'connector_params': {'param': 'CP2FSI'},
        'syn_weight': 1,
        'dynamics_params': 'FSI2CP.json',
    }
} # edges referenced by name

################################################################################
############################  EDGE EFFECTS EDGES  ##############################

if edge_effects:
    # These rules are for edge effect edges. They should directly mimic the connections
    # created previously, re-use the params set above. This keeps our code DRY
    virt_edges = [
        
]
edge_definitions = edge_definitions + virt_edges
########################## END EDGE EFFECTS ##############################
##########################################################################

##########################################################################
############################ GAP JUNCTIONS ###############################
# net = NetworkBuilder("cortex")
# conn = net.add_gap_junctions(source={'pop_name': 'FSI'}, target={'pop_name': 'FSI'},
#             resistance=1500, target_sections=['somatic'],
#             connection_rule=perc_conn,
#             connection_params={'p': 0.4})
# conn._edge_type_properties['sec_id'] = 0
# conn._edge_type_properties['sec_x'] = 0.9

# net = NetworkBuilder("cortex")
# conn = net.add_gap_junctions(source={'pop_name': 'LTS'}, target={'pop_name': 'LTS'},
#             resistance=1500, target_sections=['somatic'],
#             connection_rule=perc_conn,
#             connection_params={'p': 0.3})
# conn._edge_type_properties['sec_id'] = 0
# conn._edge_type_properties['sec_x'] = 0.9


##########################################################################
###############################  BUILD  ##################################

# Load synapse dictionaries
# see synapses.py - loads each json's in components/synaptic_models into a
# dictionary so the properties can be referenced in the files eg: syn['file.json'].get('property')
synapses.load()
syn = synapses.syn_params_dicts()

# Build your edges into the networks
build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn)

# Save the network into the appropriate network dir
save_networks(networks, network_dir)

# Usually not necessary if you've already built your simulation config
if False:
    build_env_bionet(
        base_dir = './',
        network_dir = network_dir,
        tstop = t_sim,
        dt = dt,
        report_vars = ['v'],
        celsius = 31.0,
        spikes_inputs=[
            ('thalamus', './input/thalamus_base.h5'),
            ('thalamus', './input/thalamus_short.h5'),
            ('thalamus', './input/thalamus_long.h5'),  # Name of population which spikes will be generated for, file
            ('Intbase', './input/Intbase.h5')
        ],
        components_dir='components',
        config_file='config.json',
        compile_mechanisms=False
    )
