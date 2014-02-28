from distributions.cRandom cimport global_rng
from sparse_counter cimport SparseCounter, SparseCounter_iterator


cdef extern from "distributions/models/dpm.hpp" namespace "distributions":
    cppclass rng_t
    ctypedef unsigned value_t
    cdef cppclass cModel "distributions::DirichletProcessMixture":
        cppclass hypers_t:
            float gamma
            float alpha
            float beta0
            vector[float] betas
        hypers_t hypers
        #cppclass value_t
        cppclass group_t
        void group_init (group_t &, rng_t &) nogil
        void group_add_data (group_t &, value_t &, rng_t &) nogil
        void group_remove_data (group_t &, value_t &, rng_t &) nogil
        void group_merge (group_t &, group_t &, rng_t &) nogil
        #value_t sample_value (group_t &, rng_t &) nogil
        float score_value (group_t &, value_t &, rng_t &) nogil
        float score_group (group_t &, rng_t &) nogil


cdef class Group:
    cdef cModel.group_t * ptr
    def __cinit__(self):
        self.ptr = new cModel.group_t()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        counts = raw['counts']
        self.dim = len(counts)
        cdef int i
        for i in xrange(self.dim):
            self.ptr.counts[i] = counts[i]
        return self

    def dump(self):
        counts = []
        cdef int i
        for i in xrange(self.dim):
            counts.append(self.ptr.counts[i])
        return {'counts': counts}


cdef class DirichletDiscrete:
    cdef cModel * ptr
    def __cinit__(self):
        self.ptr = new cModel()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        alphas = raw['alphas']
        cdef int dim = len(alphas)
        self.ptr.dim = dim
        cdef int i
        for i in xrange(dim):
            self.ptr.alphas[i] = alphas[i]
        return self

    def dump(self):
        alphas = []
        cdef int i
        for i in xrange(self.ptr.dim):
            alphas.append(self.ptr.alphas[i])
        return {'alphas': alphas}

    #-------------------------------------------------------------------------
    # Datatypes

    Group = staticmethod(lambda: Group())  # HACK nested classes in cython

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.ptr.dim
        self.ptr.group_init(group.ptr[0], global_rng)

    def group_add_data(self, Group group, int value):
        self.ptr.group_add_data(group.ptr[0], value, global_rng)

    def group_remove_data(self, Group group, int value):
        self.ptr.group_remove_data(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

    #def sample_value(self, Group group):
    #    cdef int value = self.ptr.sample_value(group.ptr[0], global_rng)
    #    return value

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, int value):
        return self.ptr.score_value(group.ptr[0], value, global_rng)

    def score_group(self, Group group):
        return self.ptr.score_group(group.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Serialization

    load_group = staticmethod(lambda raw: DirichletDiscrete.Group().load(raw))
    dump_group = staticmethod(lambda group: group.dump())
    load_model = staticmethod(lambda raw: DirichletDiscrete().load(raw))
    dump_model = staticmethod(lambda model: model.dump())


Model = DirichletDiscrete
