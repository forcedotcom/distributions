import numpy
from distributions.special import log, gammaln
from distributions.random import sample_discrete, sample_dirichlet


class DirichletDiscrete(object):
    def __init__(self):
        self.alphas = None

    @property
    def dim(self):
        return len(self.alphas)

    def load(self, raw):
        self.alphas = numpy.array(raw['alphas'], dtype=numpy.float)
        return self

    def dump(self):
        return {'alphas': self.alphas.tolist()}

    #-------------------------------------------------------------------------
    # Datatypes

    class Group(object):
        def __init__(self):
            self.counts = None

        def load(group, raw):
            group.counts = numpy.array(raw['counts'], dtype=numpy.int)
            return group

        def dump(group):
            return {'counts': group.counts.tolist()}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.counts = numpy.zeros(self.dim, dtype=numpy.int)

    def group_add_data(self, group, value):
        group.counts[value] += 1

    def group_remove_data(self, group, value):
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
    # Serialization

    load_group = staticmethod(lambda raw: DirichletDiscrete.Group().load(raw))
    dump_group = staticmethod(lambda group: group.dump())
    load_model = staticmethod(lambda raw: DirichletDiscrete().load(raw))
    dump_model = staticmethod(lambda model: model.dump())


Model = DirichletDiscrete
