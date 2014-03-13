cimport numpy
numpy.import_array()
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from distributions.lp.random cimport rng_t, global_rng
from distributions.lp.vector cimport VectorFloat
from distributions.mixins import ComponentModel, Serializable

cpdef int MAX_DIM = 256


ctypedef uint32_t count_t
ctypedef int Value


cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    cdef cppclass Model_cc "distributions::DirichletDiscrete<256>":
        int dim
        float alphas[256]
        #cppclass Value
        cppclass Group:
            count_t count_sum
            count_t counts[]
        cppclass Sampler:
            float ps[256]
        cppclass Scorer:
            float alpha_sum
            float alphas[256]
        cppclass Classifier:
            vector[Group] groups
            float alpha_sum
            vector[VectorFloat] scores
            VectorFloat scores_shift
        void group_init (Group &, rng_t &) nogil
        void group_add_value (Group &, Value &, rng_t &) nogil
        void group_remove_value (Group &, Value &, rng_t &) nogil
        void group_merge (Group &, Group &, rng_t &) nogil
        void sampler_init (Sampler &, Group &, rng_t &) nogil
        Value sampler_eval (Sampler &, rng_t &) nogil
        Value sample_value (Group &, rng_t &) nogil
        float score_value (Group &, Value &, rng_t &) nogil
        float score_group (Group &, rng_t &) nogil
        void classifier_init (Classifier &) nogil
        void classifier_add_group (Classifier &, rng_t &) nogil
        void classifier_remove_group (Classifier &, size_t) nogil
        void classifier_add_value (Classifier &, size_t, Value &) nogil
        void classifier_remove_value (Classifier &, size_t, Value &) nogil
        void classifier_score_value (Classifier &, Value &, float *) nogil

cdef class Group:
    cdef Model_cc.Group * ptr
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.ptr = new Model_cc.Group()
        self.dim = 0
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        counts = raw['counts']
        self.dim = len(counts)
        self.ptr.count_sum = 0
        cdef int i
        for i in xrange(self.dim):
            self.ptr.count_sum += counts[i]
            self.ptr.counts[i] = counts[i]

    def dump(self):
        counts = []
        cdef int i
        for i in xrange(self.dim):
            counts.append(self.ptr.counts[i])
        return {'counts': counts}


cdef class Classifier:
    cdef Model_cc.Classifier * ptr
    def __cinit__(self):
        self.ptr = new Model_cc.Classifier()
    def __dealloc__(self):
        del self.ptr

    def clear(self):
        self.ptr.groups.clear()

    def append(self, Group group):
        self.ptr.groups.push_back(group.ptr[0])


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

    def group_add_value(self, Group group, Value value):
        self.ptr.group_add_value(group.ptr[0], value, global_rng)

    def group_remove_value(self, Group group, Value value):
        self.ptr.group_remove_value(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

    def sample_value(self, Group group):
        cdef Value value = self.ptr.sample_value(group.ptr[0], global_rng)
        return value

    def sample_group(self, int size):
        cdef Group group = Group()
        cdef Model_cc.Sampler sampler
        self.ptr.sampler_init(sampler, group.ptr[0], global_rng)
        cdef list result = []
        cdef int i
        cdef Value value
        for i in xrange(size):
            value = self.ptr.sampler_eval(sampler, global_rng)
            result.append(value)
        return result

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, Value value):
        return self.ptr.score_value(group.ptr[0], value, global_rng)

    def score_group(self, Group group):
        return self.ptr.score_group(group.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Classification

    def classifier_init(self, Classifier classifier):
        self.ptr.classifier_init(classifier.ptr[0])

    def classifier_add_group(self, Classifier classifier):
        self.ptr.classifier_add_group(classifier.ptr[0], global_rng)

    def classifier_remove_group(self, Classifier classifier, int groupid):
        self.ptr.classifier_remove_group(classifier.ptr[0], groupid)

    def classifier_add_value(
            self,
            Classifier classifier,
            int groupid,
            Value value):
        self.ptr.classifier_add_value(classifier.ptr[0], groupid, value)

    def classifier_remove_value(
            self,
            Classifier classifier,
            int groupid,
            Value value):
        self.ptr.classifier_remove_value(classifier.ptr[0], groupid, value)

    def classifier_score_value(
            self,
            Classifier classifier,
            Value value,
            numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == classifier.ptr.groups.size(), \
            "scores_accum != len(classifier)"
        cdef float * data = <float *> scores_accum.data
        self.ptr.classifier_score_value(classifier.ptr[0], value, data)

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alphas': [1.0, 4.0]},
            'values': [0, 1, 1, 1, 1, 0, 1],
        },
        {
            'model': {'alphas': [0.5, 0.5, 0.5, 0.5]},
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletDiscrete(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group


Model = DirichletDiscrete
