from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from distributions.cRandom cimport global_rng
from distributions.sparse_counter cimport SparseCounter
from distributions.mixins import Serializable


cdef extern from "distributions/models/dpm.hpp" namespace "distributions":
    cppclass rng_t
    ctypedef unsigned value_t
    cdef cppclass Model_cc "distributions::DirichletProcessMixture":
        cppclass hypers_t:
            float gamma
            float alpha
            float beta0
            vector[float] betas
        hypers_t hypers
        #cppclass value_t
        cppclass group_t:
            SparseCounter counts
        cppclass sampler_t:
            vector[float] probs
        cppclass scorer_t:
            vector[float] scores
        void group_init (group_t &, rng_t &) nogil
        void group_add_value (group_t &, value_t &, rng_t &) nogil
        void group_remove_value (group_t &, value_t &, rng_t &) nogil
        void group_merge (group_t &, group_t &, rng_t &) nogil
        void sampler_init (sampler_t &, group_t &, rng_t &) nogil
        value_t sampler_eval (sampler_t &, rng_t &) nogil
        value_t sample_value (group_t &, rng_t &) nogil
        float score_value (group_t &, value_t &, rng_t &) nogil
        float score_group (group_t &, rng_t &) nogil


cdef class Group:
    cdef Model_cc.group_t * ptr
    def __cinit__(self):
        self.ptr = new Model_cc.group_t()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        cdef SparseCounter * counts = & self.ptr.counts
        counts.clear()
        for i, count in raw['counts'].iteritems():
            counts.init_count(int(i), count)

    def dump(self):
        counts = {}
        cdef SparseCounter.iterator it = self.ptr.counts.begin()
        cdef SparseCounter.iterator end = self.ptr.counts.end()
        while it != end:
            counts[str(deref(it).first)] = deref(it).second
            inc(it)
        return {'counts': counts}


cdef class Model_cy:
    cdef Model_cc * ptr
    def __cinit__(self):
        self.ptr = new Model_cc()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        cdef Model_cc.hypers_t * hypers = & self.ptr.hypers
        hypers.gamma = float(raw['gamma'])
        hypers.alpha = float(raw['alpha'])
        hypers.beta0 = float(raw['beta0'])
        hypers.betas.clear()
        for beta in raw['betas']:
            hypers.betas.push_back(float(beta))

    def dump(self):
        cdef Model_cc.hypers_t * hypers = & self.ptr.hypers
        betas = []
        cdef int i
        cdef int size = hypers.betas.size()
        for i in xrange(size):
            betas.append(float(hypers.betas[i]))
        return {
            'gamma': float(hypers.gamma),
            'alpha': float(hypers.alpha),
            'beta0': float(hypers.beta0),
            'betas': betas,
        }

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        self.ptr.group_init(group.ptr[0], global_rng)

    def group_add_value(self, Group group, int value):
        self.ptr.group_add_value(group.ptr[0], value, global_rng)

    def group_remove_value(self, Group group, int value):
        self.ptr.group_remove_value(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

    def realize(self):
        raise NotImplementedError()

    def sample_value(self, Group group):
        cdef int value = self.ptr.sample_value(group.ptr[0], global_rng)
        return value

    def sample_group(self, int size):
        cdef Group group = Group()
        cdef Model_cc.sampler_t sampler
        self.ptr.sampler_init(sampler, group.ptr[0], global_rng)
        cdef list result = []
        cdef int i
        cdef int value
        for i in xrange(size):
            value = self.ptr.sampler_eval(sampler, global_rng)
            result.append(value)
        return result

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, int value):
        return self.ptr.score_value(group.ptr[0], value, global_rng)

    def score_group(self, Group group):
        return self.ptr.score_group(group.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {
                'gamma': 0.5,
                'alpha': 0.5,
                'beta0': 0.1,
                'betas': [0.5, 0.5, 0.5],
            },
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletProcessMixture(Model_cy, Serializable):
    Value = int
    Group = Group


Model = DirichletProcessMixture
