from distributions.dbg.special import log, factorial, gammaln
from distributions.dbg.random import sample_gamma, sample_poisson
from distributions.mixins import ComponentModel, Serializable


class GammaPoisson(ComponentModel, Serializable):
    def __init__(self):
        self.alpha = None
        self.beta = None

    def load(self, raw):
        self.alpha = float(raw['alpha'])
        self.beta = float(raw['beta'])

    def dump(self):
        return vars(self)

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    class Group(object):
        def __init__(self):
            self.count = None
            self.sum = None
            self.log_prod = None

        def load(self, raw):
            self.count = int(raw['count'])
            self.sum = int(raw['sum'])
            self.log_prod = float(raw['log_prod'])

        def dump(self):
            return vars(self)

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.count = 0
        group.sum = 0
        group.log_prod = 0.

    def group_add_value(self, group, value):
        group.count += 1
        group.sum += int(value)
        group.log_prod += log(factorial(value))

    def group_remove_value(self, group, value):
        group.count -= 1
        group.sum -= int(value)
        group.log_prod -= log(factorial(value))

    def group_merge(self, destin, source):
        destin.count += source.count
        destin.sum += source.sum
        destin.log_prod += source.log_prod

    def plus_group(self, group):
        post = self.__class__()
        post.alpha = self.alpha + group.sum
        post.beta = 1. / (group.count + 1. / self.beta)
        return post

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_create(self, group=None):
        z = self if group is None else self.plus_group(group)
        return sample_gamma(z.alpha, z.beta)

    def sampler_eval(self, sampler):
        return sample_poisson(sampler)

    def sample_value(self, group):
        sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, size):
        sampler = self.sampler_create()
        return [self.sampler_eval(sampler) for _ in xrange(size)]

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, group, value):
        z = self.plus_group(group)
        return gammaln(z.alpha + value) - gammaln(z.alpha) - \
            z.alpha * log(z.beta) + \
            (z.alpha + value) * log(1. / (1. + 1. / z.beta)) - \
            log(factorial(value))

    def score_group(self, group):
        z = self.plus_group(group)
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


Model = GammaPoisson
