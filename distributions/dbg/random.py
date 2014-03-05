from math import log, pi, sqrt
import numpy.random
from numpy.random.mtrand import dirichlet as sample_dirichlet
from numpy import dot, inner
from numpy.linalg import cholesky, det, inv
from numpy.random import multivariate_normal, beta
from scipy.stats import chi2
from scipy.special import gammaln
from distributions.util import scores_to_probs
import logging


LOG = logging.getLogger(__name__)


# pacify pyflakes
assert sample_dirichlet


def seed(x):
    numpy.random.seed(x)
    try:
        import distributions.cRandom
        distributions.cRandom.seed(x)
    except ImportError:
        pass


def sample_discrete_log(scores):
    probs = scores_to_probs(scores)
    return sample_discrete(probs, total=1.0)


def sample_discrete(probs, total=None):
    """
    Draws from a discrete distribution with the given (possibly unnormalized)
    probabilities for each outcome.

    Returns an int between 0 and len(probs)-1, inclusive
    """
    if total is None:
        total = float(sum(probs))
    dart = numpy.random.rand() * total
    running = 0.0
    for i, prob in enumerate(probs):
        running += prob
        if running >= dart:
            return i
    LOG.error(
        'imprecision in sample_discrete',
        dict(total=total, dart=dart, probs=probs))
    raise ValueError('imprecision in sample_discrete')


def sample_student_t(dof, mu, Sigma):
    p = len(mu)
    x = numpy.random.chisquare(dof, 1)
    z = numpy.random.multivariate_normal(numpy.zeros(p), Sigma, (1,))
    return (mu + z / numpy.sqrt(x))[0]


def score_student_t(x, n, mu, sigma):
    p = len(mu)
    z = x - mu
    S = inner(inner(z, inv(sigma)), z)
    score = (
        gammaln(0.5 * (n + p))
        - gammaln(0.5 * n)
        - 0.5 * (p * log(n * pi) + log(det(sigma)) + (n + p) * log(1 + S / n))
    )
    return score


def sample_wishart_naive(nu, Lambda):
    """
    From the definition of the Wishart
    Runs in linear time
    """
    d = Lambda.shape[0]
    X = multivariate_normal(mean=numpy.zeros(d), cov=Lambda, size=nu)
    S = numpy.dot(X.T, X)
    return S


def sample_wishart(nu, Lambda):
    ch = cholesky(Lambda)
    d = Lambda.shape[0]
    z = numpy.zeros((d, d))
    for i in xrange(d):
        if i != 0:
            z[i, :i] = numpy.random.normal(size=(i,))
        z[i, i] = sqrt(numpy.random.gamma(0.5 * nu - d + 1, 2.0))
    return dot(dot(dot(ch, z), z.T), ch.T)


def sample_wishart_v2(nu, Lambda):
    """
    From Sawyer, et. al. 'Wishart Distributions and Inverse-Wishart Sampling'
    Runs in constant time
    Untested
    """
    d = Lambda.shape[0]
    ch = cholesky(Lambda)
    T = numpy.zeros((d, d))
    for i in xrange(d):
        if i != 0:
            T[i, :i] = numpy.random.normal(size=(i,))
        T[i, i] = sqrt(chi2.rvs(nu - i + 1))
    return dot(dot(dot(ch, T), T.T), ch.T)


def sample_partition_from_counts(items, counts):
    """
    Sample a partition of a list of items, as a lists of lists that satisfies
    the group sizes in counts.
    """
    assert sum(counts) == len(items), 'counts do not sum to item count'
    order = numpy.random.permutation(len(items))
    i = 0
    partition = []
    for k in range(len(counts)):
        partition.append([])
        for j in range(counts[k]):
            partition[-1].append(items[order[i]])
            i += 1
    return partition


def sample_stick(gamma, tol=1e-3):
    """
    Truncated sample from a dirichlet process using stick breaking
    """
    betas = []
    Z = 0.
    while 1 - Z > tol:
        new_beta = (1 - Z) * beta(1., gamma)
        betas.append(new_beta)
        Z += new_beta
    return {i: b / Z for i, b in enumerate(betas)}
