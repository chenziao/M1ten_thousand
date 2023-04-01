import numpy as np
from numpy import random
import pandas as pd

all_synapses = pd.DataFrame([], columns=['source_gid', 'target_gid'])

##############################################################################
############################## CONNECT CELLS #################################

def one_to_one(source, target):
    sid = source.node_id
    tid = target.node_id
    nsyn = 1 if sid == tid else None
    return nsyn

def one_to_one_offset(source, target, offset=0):
    sid = source.node_id
    tid = target.node_id - offset
    nsyn = 1 if sid == tid else None
    return nsyn

# TODO: use one_to_all iterator
def one_to_one_thal(source, target, offset):
    sid = source.node_id
    tid = target.node_id
    if sid >= 4000:
        tid = tid - offset
    nsyn = 1 if sid == tid else None
    return nsyn

# TODO: use one_to_all iterator
def one_to_one_intbase(source, target, offset1, offset2):
    sid = source.node_id
    tid = target.node_id
    if sid < 1000:
        tid = tid - offset1
    elif sid >= 1000:
        tid = tid - offset2
    nsyn = 1 if sid == tid else None
    return nsyn

def euclid_dist(p1, p2):
    """Euclidean distance between two points (coordinates in numpy)"""
    dvec = p1 - p2
    return np.sqrt(dvec @ dvec)

SYN_MIN_DELAY = 0.8 # ms
SYN_VELOCITY = 1.0 # mm/ms
def syn_dist_delay_feng(source, target):
    dist = euclid_dist(target['positions'], source['positions']) / 1000 # um to mm
    del_fluc = 0.05 * random.randn()
    delay = max(dist / SYN_VELOCITY + SYN_MIN_DELAY + del_fluc, 0.)
    return delay

# TODO: select based on cell types of a pair
TARGET_SEC_ID = {
        'CP': 1, 'CS': 1,
        'CTH': 1, 'CC': 1,
        'FSI': 0, 'LTS': 0
    }
def get_target_sec_id(source, target):
    # 0 - Target soma, 1 - Target basal dendrite
    sec_id = TARGET_SEC_ID.get(source['pop_name'], None)
    if sec_id is None:
        if source['model_type'] == 'virtual':
            sec_id = 1
            print('virtual cell') # Testing
        else: # We really don't want a default case so we can catch errors
            # return 0
            import pdb; pdb.set_trace()
    return sec_id

def syn_dist_delay_feng_section(source, target, sec_id=None, sec_x=0.9):
    if sec_id is None: # allows for overriding
        sec_id = get_target_sec_id(source, target)
    return syn_dist_delay_feng(source, target), sec_id, sec_x

def syn_uniform_delay_section(source, target, sec_id=None, sec_x=0.9, low=0.5, high=1):
    if sec_id is None: # allows for overriding
        sec_id = get_target_sec_id(source, target)
    return random.uniform(low, high), sec_id, sec_x


def points_in_cylinder(pt1, pt2, r, q):
    # https://stackoverflow.com/questions/47932955/how-to-check-if-a-3d-point-is-inside-a-cylinder
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    c1 = np.dot(q - pt1, vec) >= 0
    c2 = np.dot(q - pt2, vec) <= 0
    c3 = np.linalg.norm(np.cross(q - pt1, vec),axis=1) <= const

    return c1 & c2 & c3
    #return np.where(np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec),axis=1) <= const)

def gaussian(x, mean=0., std=1., pmax=1.):
    x = (x - mean) / std
    return pmax * np.exp(x * x / 2)

def decision(prob, size=None):
    """
    Make random decision based on input probability.
    prob: scalar input
    Return bool array if size specified, otherwise scalar
    """
    return (random.rand() if size is None else random.rand(size)) < prob

def decisions(prob):
    """
    Make random decision based on input probabilities.
    prob: iterable
    Return bool array of the same shape
    """
    prob = np.asarray(prob)
    return random.rand(*prob.shape) < prob

def sphere_dropoff(source, targets, p, stdev, track_list=None, no_recip=False, warn=False):
    # This Function is different from the one above by adding a smooth probability drop-off in a sphere.
    # p is the max probability which will occur at a distance of 0. Also supply a standard devation.
    """
    track_list: supply a list to append and track the synapses with
    one_to_all iterator for increased speed.
    returns a list of len(targets) where the value is the number of synapses at the index=cellid
    if no_recip is set to true, any connection that has been made previously and is reciprical won't be considered
    """
    global all_synapses
    sid = source.node_id
    tids = np.array([t.node_id for t in targets])

    # List of number of synapses that will be returned, initialized to 0
    syns = np.zeros(tids.size, dtype=np.int8)

    # Get all existing targets for that source that can't be targetted here
    existing = all_synapses[all_synapses.source_gid==sid]
    existing_list = existing[existing.target_gid.isin(tids)].target_gid.tolist()

    # indices of available targets ignoring existing synapses
    available = np.nonzero(~np.isin(tids, existing_list))[0]
    if no_recip:
        recur = [i['source_gid'] for i in track_list if i['target_gid']==sid]
        available = available(~np.isin(tids[available], recur))

    # Get the distances between all available targets and the source
    src_pos = np.asarray(source['positions'])
    dist = np.array([euclid_dist(target['positions'], src_pos)
                     for target in np.asarray(targets)[available]])
    # get probabilities according to the distance it is away from the cell.
    probs = gaussian(dist, 0, stdev, p)

    # Now we chose among all tids, based on the probabilities we got 
    # from the gaussian function whether that cell gets connected or not.
    syn_idx = available[decisions(probs)]
    syns[syn_idx] = 1
    chosen = tids[syn_idx]
    
    #Add to lists
    new_syns = pd.DataFrame(chosen, columns=['target_gid'])
    new_syns['source_gid'] = sid
    all_synapses = all_synapses.append(new_syns, ignore_index=True)

    if track_list is not None:
        for target in chosen:
            track_list.append({'source_gid': sid,'target_gid': target})
    return syns

def syn_percent(source, target, p, track_list=None):
    """
    track_list: supply a list to append and track the synapses with
    """
    global all_synapses

    sid = source.node_id
    tid = target.node_id
    # No autapses
    if sid==tid:
        return None

    # Do not add synapses if they already exist, we don't want duplicates
    if ((all_synapses.source_gid == sid) & (all_synapses.target_gid == tid)).any():
        return None

    if decision(p):
        all_synapses = all_synapses.append(
            {'source_gid': sid, 'target_gid': tid}, ignore_index=True)
        if track_list is not None:#we only want to track synapses that may have a recurrent connection, will speed up build time considerably
            track_list.append({'source_gid':source['node_id'],'target_gid':target['node_id']})
        return 1
    else:
        return 0

def syn_percent_o2a(source, targets, p, track_list=None,
                    no_recip=False, min_dist=0, max_dist=300,
                    angle_dist=False, angle_dist_radius=100, warn=False):
    """
    track_list: supply a list to append and track the synapses with
    one to all connector for increased speed.
    returns a list of len(targets) where the value is the number of synapses at the index=cellid
    if no_recip is set to true, any connection that has been made previously and is reciprical won't be considered
    """
    original_p = p

    global all_synapses
    sid = source.node_id
    tids = np.array([target.node_id for target in targets])

    #List of synapses that will be returned, initialized to 0 synapses
    syns = np.array([0 for _ in range(len(targets))])

    #Get all existing targets for that source that can't be targetted here
    existing = all_synapses[all_synapses.source_gid == sid]
    existing_list = existing[existing.target_gid.isin(tids)].target_gid.tolist()

    #remove existing synapses from available list
    available = tids.copy()
    available = available[~np.isin(available,existing_list)]
    if no_recip:
        recur = [i['source_gid'] for i in track_list if i['target_gid'] == sid]
        available = available[~np.isin(available,recur)]

    dist = None
    mask_dist = None

    if source.get('positions') is not None:
        src_pos = np.array(source['positions'])
        trg_pos = np.array([target['positions'] for target in targets])
        #if angle_dist: #https://github.com/latimerb/SPWR_BMTK2/blob/master/build_network.py#L148-L176
        #    """
        #    'finding the perpendicular distance from a three dimensional vector ... the goal was simply
        #     to calculate the perpendicular distance of the target cell from the source cell’s direction vector...
        #     the Euclidean distance would be the hypotenuse of that right triangle so the
        #     perpendicular distance should be the opposite side.
        #     the way I was thinking about it was to imagine a cylinder with its center around the [directional] vector
        #     ... and only cells that fall in the cylinder are eligible for connection' - Per Ben
        #    """
        #    vec_pos = np.array([np.cos(src_angle_x), np.sin(src_angle_y), np.sin(src_angle_x)])
        #    dist = np.linalg.norm(np.cross((trg_pos - src_pos), (trg_pos - vec_pos)),axis=1) / np.linalg.norm((vec_pos - src_pos))
        if angle_dist:
            # Checks if points are inside a cylinder
            src_angle_x = np.array(source['rotation_angle_zaxis'])
            src_angle_y = np.array(source['rotation_angle_yaxis'])

            vec_pos = np.array([np.cos(src_angle_x), np.sin(src_angle_y), np.sin(src_angle_x)])
            pt1 = src_pos + vec_pos * min_dist
            pt2 = src_pos + vec_pos * max_dist # Furthest point (max dist away from position of cell)

            mask_dist = points_in_cylinder(pt1, pt2, angle_dist_radius, trg_pos)

        else:
            dist = np.linalg.norm(trg_pos - src_pos,axis=1)
            mask_dist = np.array((dist < max_dist) & (dist > min_dist))

        # Since we cut down on the number of available cells due to distance constraints we need to scale up the p
        avg_connected = p*len(targets)
        # new p
        num_available = len(np.where(mask_dist==True)[0])
        if num_available:
            p = avg_connected/num_available
        else:
            p = 1.1

        if p > 1:
            p = 1
            if not angle_dist and warn:
                sorted_dist = np.sort(dist)
                minimum_max_dist = sorted_dist[int(avg_connected)]
                print("Warning: distance constraint (max_dist:" + str(max_dist) + ") between gid " + str(sid) + " and target gids " +
                    str(min(tids)) + "-" + str(max(tids)) + " prevented " + str(original_p*100) + "% overall connectivity. " +
                    "To achive this connectivity, max_dist would have needed to be greater than " + str(minimum_max_dist))

    # of those remaining we want p% chosen
    n = int(len(tids)*p)
    extra = 1 if decision(p * 100 - np.floor(p * 100)) else 0
    n = n + extra #This will account for half percentages

    if len(available) < n:
        n = len(available)

    chosen = random.choice(available,size=n,replace=False)
    mask = np.isin(tids,chosen)

    if mask_dist is not None:
        mask = mask & mask_dist
        chosen = np.where(mask==True)[0]

    syns[mask] = 1

    #Add to lists
    new_syns = pd.DataFrame(chosen,columns=['target_gid'])
    new_syns['source_gid'] = sid
    all_synapses = all_synapses.append(new_syns,ignore_index=True)

    if track_list is not None:
        #track_list = track_list.append(new_syns,ignore_index=True)
        for target in chosen:
            track_list.append({'source_gid':sid,'target_gid':target})
    #any index selected will be set to 1 and returned
    return syns

def mc_lateral_gaussian_cylinder(source,targets,p,stdev,track_list=None,no_recip=False, warn=False):
    # A gaussian distribution for the connection probability is fit given standard deviation and the max connection probability only in the horizontal.
    # This function is specifically for a motor cortex model that has 2 layers each 250 um thick.
    # No pyr to other cell connections outside the layer except CP2CS and CS2CP connections which will be able to traverse both layers from top to bottom
    # I was really lazy to access whether or not the connection was of the type mentioned above because of the weird numpy for loop thing so
    # I just copied over the function and removed the same-layer condition. The CP2CS and CS2CP function is the next one
    """
    track_list: supply a list to append and track the synapses with
    one to all connector for increased speed.
    returns a list of len(targets) where the value is the number of synapses at the index=cellid
    if no_recip is set to true, any connection that has been made previously and is reciprical won't be considered
    """
    global all_synapses
    sid = source.node_id
    tids = np.array([target.node_id for target in targets])

    #List of synapses that will be returned, initialized to 0 synapses
    syns = np.array([0 for _ in range(len(targets))])

    #Get all existing targets for that source that can't be targetted here
    existing = all_synapses[all_synapses.source_gid == sid]
    existing_list = existing[existing.target_gid.isin(tids)].target_gid.tolist()

    #remove existing synapses from available list
    available = tids.copy()
    available = available[~np.isin(available,existing_list)]
    if no_recip:
        recur = [i['source_gid'] for i in track_list if i['target_gid'] == sid]
        available = available[~np.isin(available,recur)]

    # Filter out only same layer in available
    mask_same_layer=[]
    if source.get('positions') is not None:
        mask_same_layer= np.array([target.node_id for target in targets if (source['positions'][-1] < 250 and target['positions'][-1] < 250 ) or (source['positions'][-1] >=250 and target['positions'][-1] >= 250 )])
        #print(mask_same_layer)
        available = available[~np.isin(available,mask_same_layer)]
    # Get the whole available targets from the available tids that we just found
    mask = np.isin(tids,available)
    available_targets = np.array(targets)[mask]
    # Get the distances between all available targets and the source
    dist = None
    if source.get('positions') is not None:
        src_pos = np.array(source['positions'][:-1])
        trg_pos = np.array([target['positions'][:-1] for target in available_targets])
        dist = np.linalg.norm(trg_pos - src_pos,axis=1)
    # Here we modify p according to the distance it is away from the cell.\
    probs = gaussian(dist, 0, stdev, p)
    # Now we chose among all tids, based on the probabilities we got from the gaussian distribution whether that cell gets connected or not.
    #decide_many=np.vectorize(decision)
    #synmask=decide_many(probs)
    #assign_many=np.vectorize(assign_syn_number)
    #syns=assign_many(mask, synmask)
    #cells_with_syns=np.vectorize(cells_w_syn)
    #chosen = cells_with_syns(available,synmask)
    syn_mask = []
    chosen = []
    j=0
    for i in range(len(tids)):
        if mask[i]:
            if decision(probs[j]):
                syn_mask.append(True)
                chosen.append(available[j])
            else:
                syn_mask.append(False)
            j=j+1
        else:
            syn_mask.append(False)

    syns[syn_mask] = 1

    #Add to lists
    new_syns = pd.DataFrame(chosen,columns=['target_gid'])
    new_syns['source_gid'] = sid
    all_synapses = all_synapses.append(new_syns,ignore_index=True)

    if track_list is not None:
        #track_list = track_list.append(new_syns,ignore_index=True)
        for target in chosen:
            track_list.append({'source_gid':sid,'target_gid':target})
    #any index selected will be set to 1 and returned
    return syns

def cs2cp_gaussian_cylinder(source,targets,p,stdev,track_list=None,no_recip=False, warn=False):
    # A gaussian distribution for the connection probability is fit given standard deviation and the max connection probability only in the horizontal.
    # This function is specifically for a motor cortex model that has 2 layers each 250 um thick.
    # No pyr to other cell connections outside the layer except CP2CS and CS2CP connections which will be able to traverse both layers from top to bottom
    """
    track_list: supply a list to append and track the synapses with
    one to all connector for increased speed.
    returns a list of len(targets) where the value is the number of synapses at the index=cellid
    if no_recip is set to true, any connection that has been made previously and is reciprical won't be considered
    """
    global all_synapses
    sid = source.node_id
    tids = np.array([target.node_id for target in targets])

    #List of synapses that will be returned, initialized to 0 synapses
    syns = np.array([0 for _ in range(len(targets))])

    #Get all existing targets for that source that can't be targetted here
    existing = all_synapses[all_synapses.source_gid == sid]
    existing_list = existing[existing.target_gid.isin(tids)].target_gid.tolist()

    #remove existing synapses from available list
    available = tids.copy()
    available = available[~np.isin(available,existing_list)]
    if no_recip:
        recur = [i['source_gid'] for i in track_list if i['target_gid'] == sid]
        available = available[~np.isin(available,recur)]
    # Get the whole available targets from the available tids that we just found
    mask = np.isin(tids,available)
    available_targets = np.array(targets)[mask]
    # Get the distances between all available targets and the source
    dist = None
    if source.get('positions') is not None:
        src_pos = np.array(source['positions'][:-1])
        trg_pos = np.array([target['positions'][:-1] for target in available_targets])
        dist = np.linalg.norm(trg_pos - src_pos,axis=1)
    # Here we modify p according to the distance it is away from the cell.\
    probs = gaussian(dist, 0, stdev, p)
    # Now we chose among all tids, based on the probabilities we got from the gaussian distribution whether that cell gets connected or not.
    #decide_many=np.vectorize(decision)
    #synmask=decide_many(probs)
    #assign_many=np.vectorize(assign_syn_number)
    #syns=assign_many(mask, synmask)
    #cells_with_syns=np.vectorize(cells_w_syn)
    #chosen = cells_with_syns(available,synmask)
    syn_mask = []
    chosen = []
    j=0
    for i in range(len(tids)):
        if mask[i]:
            if decision(probs[j]):
                syn_mask.append(True)
                chosen.append(available[j])
            else:
                syn_mask.append(False)
            j=j+1
        else:
            syn_mask.append(False)

    syns[syn_mask] = 1

    #Add to lists
    new_syns = pd.DataFrame(chosen,columns=['target_gid'])
    new_syns['source_gid'] = sid
    all_synapses = all_synapses.append(new_syns,ignore_index=True)

    if track_list is not None:
        #track_list = track_list.append(new_syns,ignore_index=True)
        for target in chosen:
            track_list.append({'source_gid':sid,'target_gid':target})
    #any index selected will be set to 1 and returned
    return syns

def recurrent_connector(source,target,p,all_edges=[],min_syn=1, max_syn=1):
    """
    General logic:
    1. Given a *potential* source and target
    2. Look through all edges currently made
    3. If any of the current edges contains
        a. the current source as a previous target of
        b. the current target as a prevous source
    4. Return number of synapses per this connection, 0 otherwise (no connection)
    """
    global all_synapses

    sid = source.node_id
    tid = target.node_id

    # Do not add synapses if they already exist, we don't want duplicates
    if ((all_synapses.source_gid == sid) & (all_synapses.target_gid == tid)).any():
        return None

    for e in all_edges: #should probably make this a pandas df to speed up building... and use .any() to search
        if sid == e['target_gid'] and tid == e['source_gid']:
            #print('found recurrent')

            if decision(p):
                #print('--------------connecting')
                all_synapses = all_synapses.append({'source_gid':sid,'target_gid':tid},ignore_index=True)
                return random.randint(min_syn, max_syn)
            else:
                return 0
    return 0

def recurrent_connector_o2a(source,targets,p,all_edges=[],min_syn=1,max_syn=1):

    global all_synapses
    sid = source.node_id
    tids = np.array([target.node_id for target in targets])

    #List of synapses that will be returned, initialized to 0 synapses
    syns = np.array([0 for _ in range(len(targets))])

    existing = all_synapses[all_synapses.source_gid == sid]
    existing_list = existing[existing.target_gid.isin(tids)].target_gid.tolist()

    #remove existing synapses from available list
    available = tids.copy()
    available = available[~np.isin(available,existing_list)]

    #remove any connection that is not in the all_edges list from 'available' list
    recur = [i['source_gid'] for i in all_edges if i['target_gid'] == sid]
    available = available[np.isin(available,recur)]
    #import pdb; pdb.set_trace()

    # of those remaining we want p% chosen
    n = int(len(available)*p)
    extra = 1 if decision(p * 100 - np.floor(p * 100)) else 0
    n = n + extra #This will account for half percentages
    if available != []:
        chosen = random.choice(available,size=n,replace=False)
    else:
        chosen = []

    syns[np.isin(tids,chosen)] = 1

    #Add to lists
    new_syns = pd.DataFrame(chosen,columns=['target_gid'])
    new_syns['source_gid'] = sid
    all_synapses = all_synapses.append(new_syns,ignore_index=True)

    #any index selected will be set to 1 and returned
    return syns

def perc_conn(source, targets, perc=0.2):
    nsyn = 1 if random.rand() < perc else None
    return nsyn

def section_id_placement():
    return 1, 0.6
