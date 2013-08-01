# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from math import exp, log, pi, sqrt
import scipy
import scipy.stats
from scipy.stats import chi2
from scipy.special import gammaln
import logging
import numpy
from numpy import dot, inner
from numpy.linalg import cholesky, det, inv
from numpy.random import permutation, multivariate_normal, beta


LOG = logging.getLogger(__name__)


fln_table = [0., 0., .69314, 1.79175, 3.17805, 4.78749,
    6.57925, 8.52516, 10.60460, 12.80182]
H_table = [0, 1.0, 1.5, 1.8333333333333333, 2.083333333333333,
    2.2833333333333332, 2.4499999999999997, 2.5928571428571425,
    2.7178571428571425, 2.8289682539682537, 2.9289682539682538,
    3.0198773448773446, 3.1032106782106781, 3.1801337551337552,
    3.2515623265623268, 3.3182289932289937, 3.3807289932289937,
    3.4395525226407582, 3.4951080781963135, 3.5477396571436821]


def seed(x):
    numpy.random.seed(x)
    try:
        import conjugate.scimath
        conjugate.scimath.seed(x)
    except ImportError:
        pass


def scores_to_probs(scores):
    scores = numpy.array(scores)
    scores -= scores.max()
    probs = numpy.exp(scores)
    probs /= probs.sum()
    return probs


def score_to_empirical_kl(score, count):
    '''
    Convert total log score to KL( empirical || model ),
    where the empirical pdf is uniform over `count` datapoints.
    '''
    count = float(count)
    return -score / count - log(count)


def discrete_draw_log(scores):
    probs = scores_to_probs(scores)
    dart = numpy.random.rand()
    total = 0.
    for i, prob in enumerate(probs):
        total += prob
        if total >= dart:
            return i
    LOG.error('imprecision in discrete_draw_log', dict(
        total=total, dart=dart, scores=scores))
    raise ValueError('imprecision in discrete_draw_log')


def discrete_draw(p):
    """
    Draws from a discrete distribution with the given (possibly unnormalized)
    probabilities for each outcome.

    Returns an int between 0 and len(p)-1, inclusive
    """
    z = float(sum(p))
    a = numpy.random.rand()
    tot = 0.0
    for i in range(len(p)):
        tot += p[i] / z
        if a < tot:
            return i
    raise ValueError('bug in discrete_draw')


def student_t_sample(df, mu, Sigma):
    p = len(mu)
    x = numpy.random.chisquare(df, 1)
    z = numpy.random.multivariate_normal(numpy.zeros(p), Sigma, (1,))
    return (mu + z / numpy.sqrt(x))[0]


def log_student_t(x, n, mu, sigma):
    p = len(mu)
    z = x - mu
    S = inner(inner(z, inv(sigma)), z)
    return gammaln(0.5 * (n + p)) - \
        (gammaln(0.5 * n) + 0.5 *
            (p * log(n) + p * log(pi) + log(det(sigma)) +
            (n + p) * log(1 + S / n)))


def wishart_sample(nu, Lambda):
    ch = cholesky(Lambda)
    d = Lambda.shape[0]
    z = numpy.zeros((d, d))
    for i in xrange(d):
        if i != 0:
            z[i, :i] = numpy.random.normal(size=(i,))
        z[i, i] = sqrt(numpy.random.gamma(0.5 * nu - d + 1, 2.0))
    return dot(dot(dot(ch, z), z.T), ch.T)


def wishart_sample2(nu, Lambda):
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


def naive_wishart_sample(nu, Lambda):
    """
    From the definition of the Wishart
    Runs in linear time
    """
    d = Lambda.shape[0]
    X = multivariate_normal(mean=numpy.zeros(d), cov=Lambda, size=nu)
    S = numpy.dot(X.T, X)
    return S


def H(J):
    """
    Return the ith harmonic number
    """
    if J <= 0:
        return 0
    elif J < 20:
        return H_table[J]
    else:
        return sum([1.0 / (j + 1.) for j in range(J)])


def factorialln(x):
    if x < 10:
        return fln_table[x]
    else:
        return gammaln(x + 1)


def partition_from_counts(x, counts):
    """
    Return a partition of x, a list of ids, as a lists of lists that satisfies
    the group sizes in counts.
    """
    N = sum(counts)
    order = permutation(N)
    i = 0
    partition = []
    for k in range(len(counts)):
        partition.append([])
        for j in range(counts[k]):
            partition[-1].append(x[order[i]])
            i += 1
    return partition


def stick(gamma, tol=1e-3):
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


def multinomial_goodness_of_fit(probs, counts, total_count, truncated=False):
    '''
    Returns p-value of truncated multinomial sample,
    using a pearson chi^2 test
    http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    '''
    chi_squared = 0
    dof = 0
    assert len(probs) == len(counts)
    for p, c in zip(probs, counts):
        if p == 1:
            return 1 if c == total_count else 0
        assert p < 1, 'bad probability: %g' % p
        if p > 0:
            mean = total_count * p
            variance = total_count * p * (1 - p)
            assert variance > 1, 'WARNING goodness of fit is inaccurate; '\
                    'use more samples'
            chi_squared += (c - mean) ** 2 / variance
            dof += 1
        else:
            print 'WARNING zero probability in goodness-of-fit test'
            if c > 0:
                return float('inf')

    if not truncated:
        dof -= 1

    survival = scipy.stats.chi2.sf(chi_squared, dof)
    return survival


def bin_samples(samples, k=10, support=[]):
    """
    Bins a collection of univariate samples into k bins of equal
    fill via the empirical cdf, to be used in goodness of fit testing.

    Returns
    counts : array k x 1
    bin_ranges : arrary k x 2

    each count is the number of samples in [bin_min, bin_max)
    except for the last bin which is [bin_min, bin_max]


    list partitioning algorithm adapted from Mark Dickinson:
    http://stackoverflow.com/questions
        /2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    """
    samples = sorted(samples)

    N = len(samples)
    q, r = divmod(N, k)
    #we need to distribute the remainder relatively evenly
    #tests will be inaccurate if we have small bins at the end
    indices = [i * q + min(r, i) for i in range(k + 1)]
    bins = [samples[indices[i]: indices[i + 1]] for i in range(k)]
    bin_ranges = []
    counts = []
    for i in range(k):
        bin_min = bins[i][0]
        try:
            bin_max = bins[i + 1][0]
        except IndexError:
            bin_max = bins[i][-1]
        bin_ranges.append([bin_min, bin_max])
        counts.append(len(bins[i]))
    if support:
        bin_ranges[0][0] = support[0]
        bin_ranges[-1][1] = support[1]
    return numpy.array(counts), numpy.array(bin_ranges)


def histogram(samples, bin_count=None):
    if bin_count == None:
        bin_count = numpy.max(samples) + 1
    v = numpy.zeros(bin_count, dtype=int)
    for sample in samples:
        v[sample] += 1
    return v
