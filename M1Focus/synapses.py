import numpy as np
import glob
import json
import os

from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model
from neuron import h

rng = np.random.default_rng(0)

TEST_NUM = 20
TEST_COUNT = 0

LOGN_PARAM_DICT = {}

def get_logn_params(m, s, sigma_lower, sigma_upper):
    key = (m, s)
    params = LOGN_PARAM_DICT.get(key)
    if params is None:
        sigma2 = np.log((s / m) ** 2 + 1)
        params = [np.log(m) - sigma2 / 2, sigma2 ** 0.5, {}]
        LOGN_PARAM_DICT[key] = params
    key = (sigma_lower, sigma_upper)
    bounds = params[2].get(key)
    if bounds is None:
        mu, sigma = params[:2]
        lower_bound = None if sigma_lower is None else np.exp(mu + sigma_lower * sigma)
        upper_bound = None if sigma_upper is None else np.exp(mu + sigma_upper * sigma)
        bounds = (lower_bound, upper_bound)
        params[2][key] = bounds
    return params[0], params[1], bounds


def gen_logn_weight(initW, stdevW, sigma_lower=None, sigma_upper=None):
    mu, sigma, bounds = get_logn_params(initW, stdevW, sigma_lower, sigma_upper)
    weight = rng.lognormal(mu, sigma)
    if bounds[0] is not None:
        weight = max(weight, bounds[0])
    if bounds[1] is not None:
        weight = min(weight, bounds[1])
    return weight


def set_syn_weight(syn, syn_params):
    initW = syn_params.get('initW')
    stdevW = syn_params.get('stdevW')
    if initW is not None:
        if stdevW:
            syn.initW = gen_logn_weight(initW, stdevW,
                sigma_lower=syn_params.get('sigma_lower_bound'),
                sigma_upper=syn_params.get('sigma_upper_bound'))
        else:
            syn.initW = initW
        if TEST_COUNT < TEST_NUM:
            print(f'Synapse initW: {initW: .4g}')
            TEST_COUNT += 1


AMPA_NMDA_STP_params = ('tau_r_AMPA', 'tau_d_AMPA', 'Use', 'Dep', 'Fac')

def AMPA_NMDA_STP(syn_params, sec_x, sec_id):
    """Create a AMPA_NMDA_STP synapse
    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    syn = h.AMPA_NMDA_STP(sec_x, sec=sec_id)
    for key in AMPA_NMDA_STP_params:
        value = syn_params.get(key)
        if value is not None:
            setattr(syn, key, value)
    set_syn_weight(syn, syn_params)
    return syn


GABA_AB_STP_params = ('tau_r_GABAA', 'tau_d_GABAA', 'e_GABAA', 'Use', 'Dep', 'Fac')

def GABA_AB_STP(syn_params, sec_x, sec_id):
    """Create a GABA_AB_STP synapse
    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    syn = h.GABA_AB_STP(sec_x, sec=sec_id)
    for key in GABA_AB_STP_params:
        value = syn_params.get(key)
        if value is not None:
            setattr(syn, key, value)
    set_syn_weight(syn, syn_params)
    return syn


def ampa_nmda_stp(syn_params, xs, secs):
    """Create a list of AMPA_NMDA_STP synapses
    :param syn_params: parameters of a synapse
    :param xs: list of normalized distances along the section
    :param secs: target sections
    :return: list of NEURON synpase objects
    """
    return [AMPA_NMDA_STP(syn_params, x, sec) for x, sec in zip(xs, secs)]


def gaba_ab_stp(syn_params, xs, secs):
    """Create a list of GABA_AB_STP synapses
    :param syn_params: parameters of a synapse
    :param xs: list of normalized distances along the section
    :param secs: target sections
    :return: list of NEURON synpase objects
    """
    return [GABA_AB_STP(syn_params, x, sec) for x, sec in zip(xs, secs)]


def load(randseed=0, rng_obj=None):
    global rng
    if rng_obj is None:
        rng = np.random.default_rng(randseed)
    else:
        rng = rng_obj
    add_synapse_model(AMPA_NMDA_STP, 'AMPA_NMDA_STP', overwrite=False)
    add_synapse_model(AMPA_NMDA_STP, overwrite=False)
    add_synapse_model(GABA_AB_STP, 'GABA_AB_STP', overwrite=False)
    add_synapse_model(GABA_AB_STP, overwrite=False)


def syn_params_dicts(syn_dir='components/synaptic_models'):
    """
    returns: A dictionary of dictionaries containing all
    properties in the synapse json files
    """
    files = glob.glob(os.path.join(syn_dir,'*.json'))
    data = {}
    for fh in files:
        with open(fh) as f:
            data[os.path.basename(fh)] = json.load(f) #data["filename.json"] = {"prop1":"val1",...}
    return data
