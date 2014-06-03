"""
Following http://www.johndcook.com/negative_binomial.pdf
The negative binomial (NB) gives the probability of seeing x
failures before the rth success given a success probability
p:
    p(x | p, r) \propto p ^ r * (1 - p) ^ x
For a given r and p, the NB has mean:
    mu = r (1 - p) / r
and variance:
    sigmasq = mu + (1 / r) * mu ** 2
"""
from distributions.dbg.special import gammaln
from distributions.dbg.random import sample_beta, sample_negative_binomial
from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'BetaNegativeBinomial'
EXAMPLES = [
    {
        'shared': {'alpha': 100., 'beta': 100., 'r': 1},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    },
]


Value = int


class Shared(SharedIoMixin):
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.r = None

    def plus_group(self, group):
        post = self.__class__()
        post.alpha = self.alpha + self.r * group.count
        post.beta = self.beta + group.sum
        return post

    def load(self, raw):
        self.alpha = raw['alpha']
        self.beta = raw['beta']
        self.r = raw['r']

    def dump(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r': self.r
        }


class Group(GroupIoMixin):
    def __init__(self):
        self.r = None
        self.count = None
        self.sum = None

    def init(self, shared):
        self.count = 0
        self.sum = 0

    def add_value(self, shared, value):
        self.count += 1
        self.sum += int(value)

    def remove_value(self, shared, value):
        self.count -= 1
        self.sum -= int(value)

    def merge(self, model, source):
        self.count += source.count
        self.sum += source.sum

    def score_value(self, shared, value):
        post = shared.plus_group(self)
        alpha = post.alpha + shared.r
        beta = post.beta + value
        score = gammaln(post.alpha + post.beta)
        score -= gammaln(alpha + beta)
        score += gammaln(alpha) - gammaln(post.alpha)
        score += gammaln(beta) - gammaln(post.beta)
        return score

    def score_data(self, shared):
        post = shared.plus_group(self)
        score = gammaln(shared.alpha + shared.beta)
        score -= gammaln(post.alpha + post.beta)
        score += gammaln(post.alpha) - gammaln(shared.alpha)
        score += gammaln(post.beta) - gammaln(shared.beta)
        return score

    def dump(self):
        return {
            'count': self.count,
            'sum': self.sum
        }

    def load(self, raw):
        self.count = int(raw['count'])
        self.sum = int(raw['sum'])


def sampler_create(shared, group=None):
    post = shared if group is None else shared.plus_group(group)
    return sample_beta(post.alpha, post.beta), shared.r


def sampler_eval(shared, sampler):
    return sample_negative_binomial(*sampler)


def sample_value(shared, group):
    sampler = sampler_create(shared, group)
    return sampler_eval(shared, sampler)


def sample_group(shared, size):
    sampler = sampler_create(shared)
    return [sampler_eval(shared, sampler) for _ in xrange(size)]
