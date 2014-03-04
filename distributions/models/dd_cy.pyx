import numpy
cimport numpy
numpy.import_array()
from distributions.cSpecial cimport log, gammaln
from distributions.cRandom cimport sample_dirichlet, sample_discrete
from distributions.mixins import ComponentModel, Serializable

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

    def dump(self):
        return {'counts': [self.counts[i] for i in xrange(self.dim)]}


cdef class Model_cy:
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

    def dump(self):
        return {'alphas': [self.alphas[i] for i in xrange(self.dim)]}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.dim
        cdef int i
        for i in xrange(self.dim):
            group.counts[i] = 0

    def group_add_value(self, Group group, int value):
        group.counts[value] += 1

    def group_remove_value(self, Group group, int value):
        group.counts[value] -= 1

    def group_merge(self, Group destin, Group source):
        cdef int i
        for i in xrange(self.dim):
            destin.counts[i] += source.counts[i]

    #-------------------------------------------------------------------------
    # Sampling

    cpdef sampler_create(self, Group group=None):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] \
            sampler = numpy.zeros(self.dim, dtype=numpy.float64)
        cdef double * ps = <double *> sampler.data
        cdef int i
        if group is None:
            for i in xrange(self.dim):
                sampler[i] = self.alphas[i]
        else:
            for i in xrange(self.dim):
                sampler[i] = group.counts[i] + self.alphas[i]
        sample_dirichlet(self.dim, ps, ps)
        return sampler

    cpdef sampler_eval(self, numpy.ndarray[numpy.float64_t, ndim=1] sampler):
        cdef double * ps = <double *> sampler.data
        return sample_discrete(self.dim, ps)

    def sample_value(self, Group group):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] \
            sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, int size):
        cdef Group group = Group()
        self.group_init(group)
        cdef numpy.ndarray[numpy.float64_t, ndim=1] \
            sampler = self.sampler_create(group)
        cdef list result = []
        cdef int i
        for i in xrange(size):
            result.append(self.sampler_eval(sampler))
        return result

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


Model = DirichletDiscrete
