from numpy.random.mtrand import dirichlet
from math import log
from scipy.special import gammaln
import numpy as np

from distributions.util import discrete_draw


DEFAULT_DIMENSIONS = 2


class model_t:
    def __init__(self, alphas):
        self.alphas = np.array(alphas, dtype=np.float)

    @property
    def dim(self):
        return len(self.alphas)


class group_t:
    def __init__(self, counts):
        self.counts = np.array(counts, dtype=np.int)


def group_load(ss=None, p=None):
    if ss is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return SS(np.zeros(D))
    else:
        return SS(ss['counts'])


def group_dump(ss):
    return {'counts': ss.counts.tolist()}


def model_load(hp=None, p=None):
    if hp is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return HP(np.ones(D, dtype=np.float32))
    else:
        return HP(np.array(hp['alphas'], dtype=np.float32))


def model_dump(hp):
    return {'alphas': hp.alphas.tolist()}


def add_data(ss, y):
    ss.counts[y] += 1


def remove_data(ss, y):
    ss.counts[y] -= 1


def sample_data(hp, ss):
    return discrete_draw(sample_post(hp, ss))


def sample_post(hp, ss):
    return dirichlet(ss.counts + hp.alphas)


def generate_post(hp, ss):
    post = sample_post(hp, ss)
    return {'p': post.tolist()}


def pred_prob(hp, ss, y):
    """
    McCallum, et. al, 'Rething LDA: Why Priors Matter' eqn 4
    """
    return log(
        (ss.counts[y] + hp.alphas[y])
        / (sum(ss.counts) + sum(hp.alphas)))


def data_prob(hp, ss):
    """
    From equation 22 of Michael Jordan's CS281B/Stat241B
    Advanced Topics in Learning and Decision Making course,
    'More on Marginal Likelihood'
    """

    a = hp.alphas
    m = ss.counts

    return sum([gammaln(a[k] + m[k]) - gammaln(a[k])
                for k in range(len(ss.counts))]) \
                + gammaln(sum(ak for ak in a)) \
                - gammaln(sum([a[k] + m[k] for k in range(len(ss.counts))]))


def add_pred_probs(hp, ss, y, scores):
    size = len(scores)
    assert len(ss) == size
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
