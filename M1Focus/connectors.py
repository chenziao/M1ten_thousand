import numpy as np
from numpy import random
from scipy.special import erf
from scipy.optimize import minimize_scalar
from functools import partial
import pandas as pd
import time

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

def syn_dist_delay_feng_section(source, target, connector=None,
                                sec_id=None, sec_x=0.9):
    if sec_id is None:
        sec_id = get_target_sec_id(source, target)
    return syn_dist_delay_feng(source, target, connector), sec_id, sec_x

def syn_uniform_delay_section(source, target, low=0.5, high=1,
                              sec_id=None, sec_x=0.9):
    if sec_id is None:
        sec_id = get_target_sec_id(source, target)
    return random.uniform(low, high), sec_id, sec_x

def section_id_placement():
    return 1, 0.6

SYN_MIN_DELAY = 0.8  # ms
SYN_VELOCITY = 1000.  # um/ms
def syn_dist_delay_feng(source, target, connector):
    if connector is None:
        dist = euclid_dist(target['positions'], source['positions'])
    else:
        dist = connector.get_conn_prop(source.node_id, target.node_id)
    del_fluc = 0.05 * random.randn()
    delay = max(dist / SYN_VELOCITY + SYN_MIN_DELAY + del_fluc, 0.)
    return delay


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


def euclid_dist(p1, p2):
    """Euclidean distance between two points (coordinates in numpy)"""
    dvec = p1 - p2
    return (dvec @ dvec) ** .5

def spherical_dist(node1, node2):
    """Spherical distance between two input nodes"""
    return euclid_dist(node1['positions'], node2['positions']).item()

def cylindrical_dist_z(node1, node2):
    """Cylindircal distance between two input nodes (ignore z-axis)"""
    return euclid_dist(node1['positions'][:2], node2['positions'][:2]).item()


NORM_COEF = (2 * np.pi) ** (-.5)  # coefficient of standard normal PDF

def gaussian(x, mean=0., stdev=1., pmax=NORM_COEF):
    """Gaussian function. Default is the PDF of normal distribution"""
    x = (x - mean) / stdev
    return pmax * np.exp(- x * x / 2)

class GaussianDropoff(object):
    """
    Connection probability is Gaussian function of the distance between cells,
    using either spherical distance or cylindrical distance.
    """
    def __init__(self, mean=0., stdev=1., min_dist=0., max_dist=np.inf,
                 pmax=1, ptotal=None, dist_type='spherical'):
        assert(min_dist >= 0 and min_dist < max_dist)
        self.min_dist = min_dist  # minimum distance for connection
        self.max_dist = max_dist  # maximum distance for connection
        self.mean = mean  # mean of Gaussian function
        self.stdev = stdev  # stdev of Gaussian function
        self.dist_type = dist_type if dist_type in ['cylindrical'] else 'spherical'
        self.ptotal = ptotal  # overall probability within distance range
        # pmax: maximum of the Gaussian function
        self.pmax = pmax if ptotal is None else self.calc_pmax_from_ptotal()
        self.set_probability_func()

    # TODO: accept convergence information
    def calc_pmax_from_ptotal(self):
        """Calculate the pmax value such that the expected overall connection
        probability to all possible targets within the distance range [r1, r2]=
        [min_dist, max_dist] equals ptotal, assuming homogeneous cell density.
        That is, integral_r1^r2 {g(r)p(r)dr} = ptotal, where g is the gaussian
        function with pmax, and p(r) is the population density at distance r.
        For cylindrical distance, p(r) = 2 * r / (r2^2 - r1^2)
        For spherical distance, p(r) = 3 * r^2 / (r2^3 - r1^3)
        The solution has a closed form, but only if resulting pmax <= 1.
        """
        pmax = self.ptotal * NORM_COEF
        mu, sig = self.mean, self.stdev
        r1, r2 = self.min_dist, self.max_dist
        x1, x2 = (r1 - mu) / sig, (r2 - mu) / sig  # normalized distance
        if self.dist_type == 'cylindrical':
            intgrl_1 = sig * mu / 2 * (erf(x2 / 2**.5) - erf(x1 / 2**.5))
            intgrl_2 = - sig * sig * (gaussian(x2) - gaussian(x1))
            pmax *= (r2**2 - r1**2) / 2 / (intgrl_1 + intgrl_2)
        else:  # self.dist_type == 'spherical'
            intgrl_1 = sig * (sig**2 + mu**2) / 2 * (erf(x2 / 2**.5) - erf(x1 / 2**.5))
            intgrl_2 = - sig * sig * ((2 * mu + sig * x2) * gaussian(x2) -
                                      (2 * mu + sig * x1) * gaussian(x1))
            pmax *= (r2 ** 3 - r1 ** 3) / 3 / (intgrl_1 + intgrl_2)
        return pmax

    def set_probability_func(self):
        """Set up function for calculating probability"""
        keys = ['mean', 'stdev', 'pmax']
        kwargs = {key: getattr(self, key) for key in keys}
        self.probability = partial(gaussian, **kwargs)

        # Verify maximum probability (not self.pmax if self.mean outside distance range)
        bounds = (self.min_dist, min(self.max_dist, 1e9))
        pmax = self.pmax if self.mean >= bounds[0] and self.mean <= bounds[1] \
            else self.probability(np.asarray(bounds)).max()
        if pmax > 1:
            d = minimize_scalar(lambda x: (self.probability(x) - 1)**2,
                                method='bounded', bounds=bounds).x
            warn = ("Warning: Maximum probability=%.3f is greater than 1. "
                    "Probability is 1 at distance %.3g.") % (pmax, d)
            if self.ptotal is not None:
                warn += " ptotal may not be reached."
            print(warn)

    def __call__(self, dist):
        """Returns correct probability within [0, 1] for single input"""
        if dist >= self.min_dist and dist <= self.max_dist:
            prob = min(self.probability(dist).item(), 1.)
        else:
            prob = 0.
        return prob

    def decisions(self, dist):
        """Return bool array of decisions given distance array"""
        dist = np.asarray(dist)
        dec = np.zeros(dist.shape, dtype=bool)
        mask = (dist >= self.min_dist) & (dist <= self.max_dist)
        dec[mask] = decisions(self.probability(dist[mask]))
        return dec


class ReciprocalConnector(object):
    def __init__(self, p0=0., p1=0., symmetric_p1=False,
                 p0_arg=None, p1_arg=None, symmetric_p1_arg=False,
                 pr=0., pr_arg=None, estimate_rho=True, rho=None,
                 n_syn0=1, n_syn1=1, autapses=False, source=None, target=None,
                 cache_data=True, verbose=True):
        self.p0, self.p0_arg = p0, p0_arg
        self.p1, self.p1_arg = p1, p1_arg
        self.symmetric_p1 = symmetric_p1 and symmetric_p1_arg
        self.symmetric_p1_arg = symmetric_p1_arg

        self.pr, self.pr_arg = pr, pr_arg
        self.estimate_rho = estimate_rho and not callable(pr) and rho is None
        self.rho = rho

        self.n_syn0, self.n_syn1 = n_syn0, n_syn1  # make sure <= 255
        self.autapses = autapses
        self.source, self.target = source, target
        self.cache = self.ConnectorCache(cache_data)
        self.verbose = verbose

        self.conn_prop = [{}, {}]  # for forward and backward stages
        self.stage = 0
        self.iter_count = 0

    # *** Two methods executed during bmtk edge creation net.add_edges() ***
    def setup_nodes(self, source=None, target=None):
        """Must run this before building connections"""
        if self.stage:
            if self.verbose:
                print("Skip adding nodes for the backward stage.")
            return  # skip if not at the initial stage

        # Update node pools
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target
        if self.source is None or len(self.source) == 0:
            raise ValueError("Source nodes do not exists")
        if self.target is None or len(self.target) == 0:
            raise ValueError("Target nodes do not exists")

        # Setup nodes
        self.source_ids = [s.node_id for s in self.source]
        self.target_ids = [t.node_id for t in self.target]
        self.n_source = len(self.source_ids)
        self.n_target = len(self.target_ids)
        self.recurrent = self.source.network_name == self.target.network_name \
            and self.n_source == self.n_target \
            and all([i == j for i, j in zip(self.source_ids, self.target_ids)])
        self.source_list = list(self.source)
        self.target_list = self.source_list if self.recurrent \
            else list(self.target)

        # Setup for recurrent connection
        if self.recurrent:
            self.p1_arg = self.p0_arg
            self.symmetric_p1_arg = True
            self.p1 = self.p0
            self.symmetric_p1 = True
            self.n_syn1 = self.n_syn0

    def edge_params(self):
        if self.stage == 0:
            params = {'iterator': 'one_to_all',
                      'connection_rule': self.make_forward_connection}
        else:
            params = {'source': self.target, 'target': self.source,
                      'iterator': 'all_to_one',
                      'connection_rule': self.make_backward_connection}
        self.stage += 1
        return params

    # *** Methods executed during bmtk network.build() ***
    # *** Helper functions ***
    class ConnectorCache(object):
        def __init__(self, enable=True):
            self.enable = enable
            self._output = {}
            self.cache_dict = {}
            self.write_mode()

        def cache_output(self, func, func_name, cache=True):
            if self.enable and cache:
                self.cache_dict[func_name] = func
                self._output[func_name] = []
                output = self._output[func_name]

                def writer(*args):
                    val = func(*args)
                    output.append(val)
                    return val
                setattr(self, func_name, writer)
            else:
                setattr(self, func_name, func)

        def write_mode(self):
            for val in self._output.values():
                val.clear()
            self.mode = 'write'
            self.iter_count = 0

        def fetch_output(self, func_name, fetch=True):
            output = self._output[func_name]

            if fetch:
                def reader(*args):
                    return output[self.iter_count]
                setattr(self, func_name, reader)
            else:
                setattr(self, func_name, self.cache_dict[func_name])

        def read_mode(self):
            if self.enable and len(self.cache_dict):
                # check whether outputs were written correctly
                output_len = [len(val) for val in self._output.values()]
                # whether any stored and have the same length
                valid = [n for n in output_len if n]
                flag = len(valid) > 0 and all(n == valid[0] for n in valid[1:])
                if flag:
                    for func_name, out_len in zip(self._output, output_len):
                        fetch = out_len > 0
                        if not fetch:
                            print("Warning: Cache did not work properly for "
                                  + func_name)
                        self.fetch_output(func_name, fetch)
                    self.iter_count = 0
                else:
                    # if output not correct, disable and use original function
                    print("Warning: Cache did not work properly.")
                    for func_name in self.cache_dict:
                        self.fetch_output(func_name, False)
                    self.enable = False
            self.mode = 'read'

        def next_it(self):
            if self.enable:
                self.iter_count += 1

    @staticmethod
    def constant_function(val):
        def constant(*arg):
            return val
        return constant

    def node_2_idx_input(self, var_func, reverse=False):
        if reverse:
            def idx_2_var(j, i):
                return var_func(self.target_list[j], self.source_list[i])
        else:
            def idx_2_var(i, j):
                return var_func(self.source_list[i], self.target_list[j])
        return idx_2_var

    def iterate_pairs(self):
        if self.recurrent:
            if self.autapses:
                for i in range(self.n_source):
                    for j in range(i, self.n_target):
                        yield i, j
            else:
                for i in range(self.n_source - 1):
                    for j in range(i + 1, self.n_target):
                        yield i, j
        else:
            for i in range(self.n_source):
                for j in range(self.n_target):
                    yield i, j

    def calc_pair(self, i, j):
        """Calculate intermediate data that can be cached"""
        cache = self.cache
        # cache = self  # test performance for not using cache
        p0_arg = cache.p0_arg(i, j)
        p1_arg = p0_arg if self.symmetric_p1_arg else cache.p1_arg(j, i)
        p0 = cache.p0(p0_arg)
        p1 = p0 if self.symmetric_p1 else cache.p1(p1_arg)
        return p0_arg, p1_arg, p0, p1

    def setup_conditional_backward_probability(self):
        # For all cases, assume p0, p1, pr are all within [0, 1] already.
        self.wrong_pr = False
        if self.rho is None:
            # Determine by pr for each pair
            if self.verbose:
                def cond_backward(cond, p0, p1, pr):
                    if p0 > 0:
                        pr_bound = (p0 + p1 - 1, min(p0, p1))
                        # check whether pr within bounds
                        if pr < pr_bound[0] or pr > pr_bound[1]:
                            self.wrong_pr = True
                            pr = min(max(pr, pr_bound[0]), pr_bound[1])
                        return pr / p0 if cond else (p1 - pr) / (1 - p0)
                    else:
                        return p1
            else:
                def cond_backward(cond, p0, p1, pr):
                    if p0 > 0:
                        pr_bound = (p0 + p1 - 1, min(p0, p1))
                        pr = min(max(pr, pr_bound[0]), pr_bound[1])
                        return pr / p0 if cond else (p1 - pr) / (1 - p0)
                    else:
                        return p1
        elif self.rho == 0:
            # Independent case
            def cond_backward(cond, p0, p1, pr):
                return p1
        else:
            # Dependent with fixed correlation coefficient rho
            def cond_backward(cond, p0, p1, pr):
                # Standard deviation of r.v. for p1
                sd = ((1 - p1) * p1) ** .5
                # Z-score of random variable for p0
                zs = ((1 - p0) / p0) ** .5 if cond else - (p0 / (1 - p0)) ** .5
                return p1 + self.rho * sd * zs
        self.cond_backward = cond_backward

    def add_conn_prop(self, src, trg, prop, stage=0):
        sid = self.source_ids[src]
        tid = self.target_ids[trg]
        conn_dict = self.conn_prop[stage]
        if stage:
            sid, tid = tid, sid  # during backward, from target to source population
        trg_dict = conn_dict.setdefault(sid, {})
        trg_dict[tid] = prop

    def get_conn_prop(self, sid, tid):
        return self.conn_prop[self.stage][sid][tid]

    # *** A sequence of major methods executed during build ***
    def setup_variables(self):
        var_set = set(('p0', 'p0_arg', 'p1', 'p1_arg',
                        'pr', 'pr_arg', 'n_syn0', 'n_syn1'))
        callable_set = set()
        # Make constant variables constant functions
        for name in var_set:
            var = getattr(self, name)
            if callable(var):
                callable_set.add(name)  # record callable variables
            else:
                setattr(self, name, self.constant_function(var))
        self.callable_set = callable_set

        # Make callable variables except a few, accept index input instead
        for name in (var_set & callable_set) - set(('p0', 'p1', 'pr')):
            var = getattr(self, name)
            setattr(self, name, self.node_2_idx_input(var, '1' in name))

    def cache_variables(self):
        # Select cacheable attrilbutes
        cache_set = set(('p0', 'p0_arg', 'p1', 'p1_arg'))
        if self.symmetric_p1:
            cache_set.remove('p1')
        if self.symmetric_p1_arg:
            cache_set.remove('p1_arg')
        # Output of callable variables will be cached
        # Constant functions will be called from cache but output not cached
        for name in cache_set:
            var = getattr(self, name)
            self.cache.cache_output(var, name, name in self.callable_set)
        if self.verbose and len(self.cache.cache_dict):
            print('Output of %s will be cached.'
                  % ', '.join(self.cache.cache_dict))

    def initialize(self):
        self.setup_variables()
        self.cache_variables()
        # Intialize connection matrix and get nubmer of pairs
        self.end_stage = 0 if self.recurrent else 1
        shape = (self.end_stage + 1, self.n_source, self.n_target)
        self.conn_mat = np.zeros(shape, dtype=np.uint8)  # 1 byte per entry

    def initial_all_to_all(self):
        if self.verbose:
            src_str, trg_str = self.get_nodes_info()
            print("\nStart building connection between: \n"
                  + src_str + "\n" + trg_str)
        self.initialize()
        cache = self.cache  # write mode

        # Estimate pr
        if self.verbose:
            self.timer = Timer('ms')
        if self.estimate_rho:
            # Make sure each cacheable function runs excatly once per iteration
            p0p1_sum = 0.
            norm_fac_sum = 0.
            n = 0
            for i, j in self.iterate_pairs():
                _, _, p0, p1 = self.calc_pair(i, j)
                p0p1 = p0 * p1
                possible = p0p1 > 0
                if possible:
                    n += 1
                    p0p1_sum += p0p1
                    norm_fac_sum += (p0 * (1 - p0) * p1 * (1 - p1)) ** .5
            if norm_fac_sum > 0:
                rho = (self.pr() * n - p0p1_sum) / norm_fac_sum
                if abs(rho) > 1:
                    print("Warning: Estimated value of rho=%.3f "
                          "outside the range [-1, 1]." % rho)
                    rho = np.clip(rho, -1, 1).item()
                    print("Force rho to be %.0f." % rho)
                elif self.verbose:
                    print("Estimated value of rho=%.3f" % rho)
                self.rho = rho
            else:
                self.rho = 0

        if self.verbose:
            self.timer.report('Time for estimating rho')

        # Setup function for calculating conditional backward probability
        self.setup_conditional_backward_probability()

        # Make connections
        cache.read_mode()
        possible_count = 0 if self.recurrent else np.zeros(3)
        for i, j in self.iterate_pairs():
            p0_arg, p1_arg, p0, p1 = self.calc_pair(i, j)
            # Check whether at all possible and count
            forward = p0 > 0
            backward = p1 > 0
            if self.recurrent:
                possible_count += forward
            else:
                possible_count += [forward, backward, forward and backward]
            # Make random decision
            if forward:
                forward = decision(p0)
            if backward:
                pr = self.pr(self.pr_arg(i, j))
                backward = decision(self.cond_backward(forward, p0, p1, pr))
            # Make connection
            if forward:
                n_forward = self.n_syn0(i, j)
                self.add_conn_prop(i, j, p0_arg, 0)
                self.conn_mat[0, i, j] = n_forward
            if backward:
                n_backward = self.n_syn1(j, i)
                if self.recurrent:
                    self.conn_mat[0, j, i] = n_backward
                    self.add_conn_prop(j, i, p1_arg, 0)
                else:
                    self.conn_mat[1, i, j] = n_backward
                    self.add_conn_prop(i, j, p1_arg, 1)
            self.cache.next_it()
        self.cache.write_mode()  # clear memory
        self.possible_count = possible_count

        if self.verbose:
            self.timer.report('Total time for creating connection matrix')
            if self.wrong_pr:
                print("Warning: Value of 'pr' outside of the bounds occurs.")
            self.connection_number_info()

    def make_connection(self):
        """ Assign number of synapses per iteration.
        Use iterator one_to_all for forward and all_to_one for backward.
        """
        nsyns = self.conn_mat[self.stage, self.iter_count, :]
        self.iter_count += 1

        # Detect end of iteration
        if self.iter_count == self.n_source:
            self.iter_count = 0
            if self.stage == self.end_stage:
                if self.verbose:
                    self.timer.report('Done! \nTime for building connections')
                else:
                    self.free_memory()
        return nsyns

    def make_forward_connection(self, source, targets, *args, **kwargs):
        # Initialize in the first iteration
        if self.iter_count == 0:
            self.stage = 0
            self.initial_all_to_all()
            if self.verbose:
                print("Assigning forward connections.")
                self.timer.start()
        return self.make_connection()

    def make_backward_connection(self, targets, source, *args, **kwargs):
        if self.iter_count == 0:
            self.stage = 1
            if self.verbose:
                print("Assigning backward connections.")
        return self.make_connection()

    def free_memory(self):
        # Do not clear self.conn_prop if it will be used by conn.add_properties
        variables = ('conn_mat', 'source_list', 'target_list',
                     'source_ids', 'target_ids')
        for var in variables:
            setattr(self, var, None)

    # *** Helper functions for verbose ***
    def get_nodes_info(self):
        source_str = self.source.network_name + ': ' + self.source.filter_str
        target_str = self.target.network_name + ': ' + self.target.filter_str
        return source_str, target_str

    def connection_number(self):
        """
        Return the number of the following:
        n_conn: connected pairs [forward, (backward,) reciprocal]
        n_poss: possible connections (prob>0) [forward, (backward, reciprocal)]
        n_pair: pairs of cells
        proportion: of connections in possible and total pairs
        """
        conn_mat = self.conn_mat.astype(bool)
        n_conn = np.count_nonzero(conn_mat, axis=(1, 2))
        n_poss = np.array(self.possible_count)
        n_pair = conn_mat.size / 2
        if self.recurrent:
            n_recp = (np.count_nonzero(conn_mat[0] & conn_mat[0].T)
                      - np.count_nonzero(np.diag(conn_mat[0]))) // 2
            n_conn = n_conn - n_recp
            n_poss = n_poss[None]
            n_pair += (1 if self.autapses else -1) * self.n_source / 2
        else:
            n_recp = np.count_nonzero(conn_mat[0] & conn_mat[1])
        n_conn = np.append(n_conn, n_recp)
        n_pair = int(n_pair)
        fraction = np.array([n_conn / n_poss, n_conn / n_pair])
        fraction[np.isnan(fraction)] = 0.
        return n_conn, n_poss, n_pair, fraction

    def connection_number_info(self):
        def arr2str(a, f):
            return ', '.join([f] * a.size) % tuple(a.tolist())
        n_conn, n_poss, n_pair, fraction = self.connection_number()
        conn_type = "(all, reciprocal)" if self.recurrent \
                    else "(forward, backward, reciprocal)"
        print("Numbers of " + conn_type + " connections:")
        print("Number of connected pairs: (%s)" % arr2str(n_conn, '%d'))
        print("Number of possible connections: (%s)" % arr2str(n_poss, '%d'))
        print("Fraction of connected pairs in possible ones: (%s)"
              % arr2str(100 * fraction[0], '%.2f%%'))
        print("Number of total pairs: %d" % n_pair)
        print("Fraction of connected pairs in all pairs: (%s)\n"
              % arr2str(100 * fraction[1], '%.2f%%'))


# Helper class
class Timer(object):
    def __init__(self, unit='sec'):
        if unit == 'ms':
            self.scale = 1e3
        elif unit == 'us':
            self.scale = 1e6
        elif unit == 'min':
            self.scale = 1 / 60
        else:
            self.scale = 1
            unit = 'sec'
        self.unit = unit
        self.start()

    def start(self):
        self._start = time.perf_counter()

    def end(self):
        return (time.perf_counter() - self._start) * self.scale

    def report(self, msg='Run time'):
        print((msg + ": %.3f " + self.unit) % self.end())


# TODO: Below are not updated or used
def perc_conn(source, targets, perc=0.2):
    nsyn = 1 if random.rand() < perc else None
    return nsyn
