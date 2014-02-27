from distributions.cRandom cimport global_rng

cpdef int MAX_DIM = 256


cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    cppclass rng_t
    ctypedef int value_t
    cdef cppclass cDirichletDiscrete "distributions::DirichletDiscrete<256>":
        int dim
        float alphas[]
        #cppclass value_t
        cppclass group_t:
            int counts[]
        void group_init (group_t & group, rng_t & rng)
        void group_add_data (group_t & group, value_t & value, rng_t & rng)
        void group_remove_data (group_t & group, value_t & value, rng_t & rng)
        void group_merge (group_t & destin, group_t & source, rng_t & rng)
        value_t sample_value (group_t & group, rng_t & rng)
        float score_value (group_t & group, value_t & value, rng_t & rng)
        float score_group (group_t & group, rng_t & rng)


cdef class Group:
    cdef cDirichletDiscrete.group_t * ptr
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.ptr = new cDirichletDiscrete.group_t()
        self.dim = 0
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
    cdef cDirichletDiscrete * ptr
    def __cinit__(self):
        self.ptr = new cDirichletDiscrete()
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

    def group_add_value(self, Group group, int value):
        self.ptr.group_add_data(group.ptr[0], value, global_rng)

    def group_remove_value(self, Group group, int value):
        self.ptr.group_remove_data(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

    def sample_value(self, Group group):
        cdef int value = self.ptr.sample_value(group.ptr[0], global_rng)
        return value

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
