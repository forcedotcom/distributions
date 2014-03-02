import numpy
from distributions.special import log, gammaln
from distributions.random import sample_discrete, sample_dirichlet
from distributions.mixins import Serializable


class DirichletDiscrete(Serializable):
    def __init__(self):
        self.alphas = None

    @property
    def dim(self):
        return len(self.alphas)

    def load(self, raw):
        self.alphas = numpy.array(raw['alphas'], dtype=numpy.float)

    def dump(self):
        return {'alphas': self.alphas.tolist()}

    #-------------------------------------------------------------------------
    # Datatypes

    class Group(object):
        def __init__(self):
            self.counts = None

        def load(self, raw):
            self.counts = numpy.array(raw['counts'], dtype=numpy.int)

        def dump(self):
            return {'counts': self.counts.tolist()}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.counts = numpy.zeros(self.dim, dtype=numpy.int)

    def group_add_value(self, group, value):
        group.counts[value] += 1

    def group_remove_value(self, group, value):
        group.counts[value] -= 1

    def group_merge(self, destin, source):
        destin.counts += source.counts

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_init(self, group):
        return sample_dirichlet(group.counts + self.alphas)

    def sampler_eval(self, sampler):
        return sample_discrete(sampler)

    def sample_value(self, group):
        sampler = self.sampler_init(group)
        return self.sampler_eval(sampler)

    def sample_group(self, size):
        group = self.Group()
        self.group_init(group)
        sampler = self.sampler_init(group)
        return [self.sampler_eval(sampler) for _ in xrange(size)]

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, group, value):
        """
        McCallum, et. al, 'Rething LDA: Why Priors Matter' eqn 4
        """
        numer = group.counts[value] + self.alphas[value]
        denom = group.counts.sum() + self.alphas.sum()
        return log(numer / denom)

    def score_group(self, group):
        """
        From equation 22 of Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """

        dim = self.dim
        a = self.alphas
        m = group.counts

        score = sum(gammaln(a[k] + m[k]) - gammaln(a[k]) for k in xrange(dim))
        score += gammaln(a.sum())
        score -= gammaln(a.sum() + m.sum())
        return score

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLE = {
        'values': [0, 1, 0, 2, 0, 1, 0],
        'model': {
            'alphas': [0.5, 0.5, 0.5, 0.5],
        },
    }


Model = DirichletDiscrete
