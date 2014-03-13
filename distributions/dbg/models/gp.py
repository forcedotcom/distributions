from distributions.dbg.special import log, factorial, gammaln
from distributions.dbg.random import sample_gamma, sample_poisson
from distributions.mixins import ComponentModel, Serializable


class GammaPoisson(ComponentModel, Serializable):
    def __init__(self):
        self.alpha = None
        self.inv_beta = None

    def load(self, raw):
        self.alpha = float(raw['alpha'])
        self.inv_beta = float(raw['inv_beta'])

    def dump(self):
        return {
            'alpha': self.alpha,
            'inv_beta': self.inv_beta,
        }

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
            return {
                'count': self.count,
                'sum': self.sum,
                'log_prod': self.log_prod,
            }

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
        post.inv_beta = self.inv_beta + group.count
        return post

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_create(self, group=None):
        post = self if group is None else self.plus_group(group)
        return sample_gamma(post.alpha, 1.0 / post.inv_beta)

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
        post = self.plus_group(group)
        return gammaln(post.alpha + value) - gammaln(post.alpha) \
            + post.alpha * log(post.inv_beta) \
            - (post.alpha + value) * log(1. + post.inv_beta) \
            - log(factorial(value))

    def score_group(self, group):
        post = self.plus_group(group)
        return gammaln(post.alpha) - gammaln(self.alpha) \
            - post.alpha * log(post.inv_beta) \
            + self.alpha * log(self.inv_beta) \
            - group.log_prod

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alpha': 1., 'inv_beta': 1.},
            'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
        }
    ]


Model = GammaPoisson
