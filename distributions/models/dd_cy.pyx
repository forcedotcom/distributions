import numpy
cimport numpy
numpy.import_array()
from distributions.cSpecial cimport log, gammaln
from distributions.cRandom cimport sample_dirichlet, sample_discrete

cpdef int MAX_DIM = 256


cdef class Group:
    cdef int counts[256]
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.dim = 0

    def load(self, raw):
        counts = raw['counts']
        self.dim = len(counts)
        assert self.dim <= MAX_DIM
        cdef int i
        for i in xrange(self.dim):
            self.counts[i] = counts[i]
        return self

    def dump(self):
        return {'counts': [self.counts[i] for i in xrange(self.dim)]}


cdef class DirichletDiscrete:
    cdef double[256] alphas
    cdef int dim
    def __cinit__(self):
        self.dim = 0

    def load(self, raw):
        alphas = raw['alphas']
        self.dim = len(alphas)
        assert self.dim <= MAX_DIM
        cdef int i
        for i in xrange(self.dim):
            self.alphas[i] = alphas[i]
        return self

    def dump(self):
        return {'alphas': [self.alphas[i] for i in xrange(self.dim)]}

    #-------------------------------------------------------------------------
    # Datatypes

    Group = staticmethod(lambda: Group())  # HACK nested classes in cython

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.dim
        cdef int i
        for i in xrange(self.dim):
            group.counts[i] = 0

    def group_add_data(self, Group group, int value):
        group.counts[value] += 1

    def group_remove_data(self, Group group, int value):
        group.counts[value] -= 1

    def group_merge(self, Group destin, Group source):
        cdef int i
        for i in xrange(self.dim):
            destin.counts[i] += source.counts[i]

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_init(self, Group group):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] \
            sampler = numpy.zeros(self.dim, dtype=numpy.float64)
        cdef double * ps = <double *> sampler.data
        cdef int i
        for i in xrange(self.dim):
            sampler[i] = group.counts[i] + self.alphas[i]
        sample_dirichlet(self.dim, ps, ps)
        return sampler

    def sampler_eval(self, numpy.ndarray[numpy.float64_t, ndim=1] sampler):
        cdef double * ps = <double *> sampler.data
        return sample_discrete(self.dim, ps)

    def sample_value(self, Group group):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] \
            sampler = self.sampler_init(group)
        return self.sampler_eval(sampler)

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, int value):
        """
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        cdef double total = 0.0
        cdef int i
        for i in xrange(self.dim):
            total += group.counts[i] + self.alphas[i]
        return log((group.counts[value] + self.alphas[value]) / total)

    def score_group(self, Group group):
        """
        From equation 22 of Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """
        cdef int i
        cdef double alpha_sum = 0.0
        cdef int count_sum = 0
        cdef double sum = 0.0
        for i in xrange(self.dim):
            alpha_sum += self.alphas[i]
        for i in xrange(self.dim):
            count_sum += group.counts[i]
        for i in xrange(self.dim):
            sum += (gammaln(self.alphas[i] + group.counts[i])
                    - gammaln(self.alphas[i]))
        return sum + gammaln(alpha_sum) - gammaln(alpha_sum + count_sum)

    #-------------------------------------------------------------------------
    # Serialization

    load_group = staticmethod(lambda raw: DirichletDiscrete.Group().load(raw))
    dump_group = staticmethod(lambda group: group.dump())
    load_model = staticmethod(lambda raw: DirichletDiscrete().load(raw))
    dump_model = staticmethod(lambda model: model.dump())


Model = DirichletDiscrete
