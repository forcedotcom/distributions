from distributions.cRandom cimport global_rng
from distributions.mixins import Serializable

cpdef int MAX_DIM = 256


cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    cppclass rng_t
    ctypedef int value_t
    cdef cppclass Model_cc "distributions::DirichletDiscrete<256>":
        int dim
        cppclass hypers_t:
            float alphas[256]
        hypers_t hypers
        #cppclass value_t
        cppclass group_t:
            int counts[]
        cppclass sampler_t:
            float ps[256]
        cppclass scorer_t:
            float alpha_sum
            float alphas[256]
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
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.ptr = new Model_cc.group_t()
        self.dim = 0
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        counts = raw['counts']
        self.dim = len(counts)
        cdef int i
        for i in xrange(self.dim):
            self.ptr.counts[i] = counts[i]

    def dump(self):
        counts = []
        cdef int i
        for i in xrange(self.dim):
            counts.append(self.ptr.counts[i])
        return {'counts': counts}


cdef class Model_cy:
    cdef Model_cc * ptr
    def __cinit__(self):
        self.ptr = new Model_cc()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        alphas = raw['alphas']
        cdef int dim = len(alphas)
        self.ptr.dim = dim
        cdef int i
        for i in xrange(dim):
            self.ptr.hypers.alphas[i] = float(alphas[i])

    def dump(self):
        alphas = []
        cdef int i
        for i in xrange(self.ptr.dim):
            alphas.append(float(self.ptr.hypers.alphas[i]))
        return {'alphas': alphas}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.ptr.dim
        self.ptr.group_init(group.ptr[0], global_rng)

    def group_add_value(self, Group group, int value):
        self.ptr.group_add_value(group.ptr[0], value, global_rng)

    def group_remove_value(self, Group group, int value):
        self.ptr.group_remove_value(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

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
                'alphas': [0.5, 0.5, 0.5, 0.5],
            },
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletDiscrete(Model_cy, Serializable):
    Value = int
    Group = Group


Model = DirichletDiscrete
