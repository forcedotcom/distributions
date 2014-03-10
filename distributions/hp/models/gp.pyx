import numpy
cimport numpy
numpy.import_array()
from distributions.hp.special cimport sqrt, log, gammaln, log_factorial
from distributions.hp.random cimport sample_poisson, sample_gamma
from distributions.mixins import ComponentModel, Serializable


ctypedef int Value


cdef class Group:
    cdef int count
    cdef int sum
    cdef double log_prod

    def load(self, raw):
        self.count = raw['count']
        self.sum = raw['sum']
        self.log_prod = raw['log_prod']

    def dump(self):
        return {
            'count': self.count,
            'sum': self.sum,
            'log_prod': self.log_prod,
        }


ctypedef double Sampler


cdef class Model_cy:
    cdef double alpha
    cdef double beta

    def load(self, raw):
        self.alpha = raw['alpha']
        self.beta = raw['beta']

    def dump(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
        }

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.count = 0
        group.sum = 0
        group.log_prod = 0.

    def group_add_value(self, Group group, int value):
        group.count += 1
        group.sum += value
        group.log_prod += log_factorial(value)

    def group_remove_value(self, Group group, int value):
        group.count -= 1
        group.sum -= value
        group.log_prod -= log_factorial(value)

    def group_merge(self, Group destin, Group source):
        destin.count += source.count
        destin.sum += source.sum
        destin.log_prod += source.log_prod

    cdef Model_cy plus_group(self, Group group):
        cdef Model_cy post = Model_cy()
        post.alpha = self.alpha + group.sum
        post.beta = 1. / (group.count + 1. / self.beta)
        return post

    #-------------------------------------------------------------------------
    # Sampling

    cpdef Sampler sampler_create(Model_cy self, Group group=None):
        cdef Model_cy z = self if group is None else self.plus_group(group)
        return sample_gamma(z.alpha, z.beta)

    cpdef Value sampler_eval(self, Sampler sampler):
        return sample_poisson(sampler)

    def sample_value(self, Group group):
        cdef Sampler sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, int size):
        cdef Sampler sampler = self.sampler_create()
        cdef list result = []
        cdef int i
        for i in xrange(size):
            result.append(self.sampler_eval(sampler))
        return result

    #-------------------------------------------------------------------------
    # Scoring

    cpdef double score_value(self, Group group, Value value):
        cdef Model_cy z = self.plus_group(group)
        return gammaln(z.alpha + value) - gammaln(z.alpha) - \
            z.alpha * log(z.beta) + \
            (z.alpha + value) * log(1. / (1. + 1. / z.beta)) - \
            log_factorial(value)

    def score_group(self, Group group):
        cdef Model_cy z = self.plus_group(group)
        return gammaln(z.alpha) - gammaln(self.alpha) + \
            z.alpha * log(z.beta) - self.alpha * log(self.beta) - \
            group.log_prod

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alpha': 1., 'beta': 1.},
            'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
        }
    ]


class GammaPoisson(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group


Model = GammaPoisson
