from libcpp.vector cimport vector
from distributions.hp.random cimport global_rng
from distributions.mixins import ComponentModel, Serializable

cpdef int MAX_DIM = 256


cdef extern from "distributions/vector.hpp" namespace "distributions":
    cppclass FloatVector:
        size_t size ()
        float & at "operator[]" (size_t index)


cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    cppclass rng_t
    ctypedef int Value
    cdef cppclass Model_cc "distributions::DirichletDiscrete<256>":
        int dim
        float alphas[256]
        #cppclass Value
        cppclass Group:
            unsigned counts[]
        cppclass Sampler:
            float ps[256]
        cppclass Scorer:
            float alpha_sum
            float alphas[256]
        cppclass VectorScorer:
            vector[FloatVector] scores
        void group_init (Group &, rng_t &) nogil
        void group_add_value (Group &, Value &, rng_t &) nogil
        void group_remove_value (Group &, Value &, rng_t &) nogil
        void group_merge (Group &, Group &, rng_t &) nogil
        void sampler_init (Sampler &, Group &, rng_t &) nogil
        Value sampler_eval (Sampler &, rng_t &) nogil
        Value sample_value (Group &, rng_t &) nogil
        float score_value (Group &, Value &, rng_t &) nogil
        float score_group (Group &, rng_t &) nogil
        void vector_scorer_init (VectorScorer &, size_t, rng_t &)
        void vector_scorer_update (VectorScorer &, size_t, Group &, rng_t &)
        void vector_scorer_eval (
                FloatVector &,
                VectorScorer &,
                Value &,
                rng_t &)


cdef class Group:
    cdef Model_cc.Group * ptr
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.ptr = new Model_cc.Group()
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


cdef class VectorScorer:
    cdef Model_cc.VectorScorer * ptr
    def __cinit__(self):
        self.ptr = new Model_cc.VectorScorer()
    def __dealloc__(self):
        del self.ptr


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
            self.ptr.alphas[i] = float(alphas[i])

    def dump(self):
        alphas = []
        cdef int i
        for i in xrange(self.ptr.dim):
            alphas.append(float(self.ptr.alphas[i]))
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
        cdef Model_cc.Sampler sampler
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

    def vector_scorer_init(self, VectorScorer scorer, int group_count):
        self.ptr.vector_scorer_init(scorer.ptr[0], group_count, global_rng)

    def vector_scorer_update(
            self,
            VectorScorer scorer,
            int group_index,
            Group group):
        self.ptr.vector_scorer_update(
                scorer.ptr[0],
                group_index,
                group.ptr[0],
                global_rng)

    def vector_scorer_eval(self, VectorScorer scorer, int value):
        cdef FloatVector scores
        self.ptr.vector_scorer_eval(scores, scorer.ptr[0], value, global_rng)
        cdef list result = []
        cdef float score
        cdef int i
        for i in xrange(scores.size()):
            score = scores.at(i)
            result.append(score)
        return result

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


class DirichletDiscrete(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group

    VectorScorer = VectorScorer


Model = DirichletDiscrete
