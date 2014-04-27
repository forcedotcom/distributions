# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
A parameter-free clustering prior based on partition entropy.

The Dirichlet prior is often used in nonparametric mixture models to model
the partitioning of observations into clusters.
This class implements a parameter-free alternative to the Dirichlet prior
that preserves exchangeability, preserves asymptotic convergence rate of
density estimation, and has an elegant interpretation as a
minimum-description-length prior.

Motivation
----------

In conjugate mixture models, samples from the posterior are sufficiently
represented by a partitioning of observations into clusters, say as an
assignment vector X with cluster labels X_i for each observation i.
In our ~10^6-observation production system, we found the data size of this
assignment vector to be a limiting factor in query latency, even after
lossless compression.  To address this problem, we tried to incorporate the
Shannon entropy of this assignment vector directly into the prior, as
an information criterion.  Surprisingly, the resulting low-entropy prior
enjoys a number of properties:

- The low-entropy prior enjoys similar asymptotic convergence as the
  Dirichlet prior.

- The probability of a clustering is elegant and easy to evaluate
  (up to an unknown normalizing constant).

- The resulting distribution resembles a CRP distribution with parameter
  alpha = exp(-1), but slightly avoids small clusters.

- MAP estimates are minimum-description-length, as measured by assignment
  vector complexity.

- The low-entropy prior is parameter free, unlike the CRP, Pitman-Yor, or
  Mixture of Finite Mixture models.

A difficulty is that the prior depends on dataset size, and is hence not a
proper nonparametric generative model.
'''

import os
from collections import defaultdict
import numpy
from numpy import log, exp
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, font_manager
import parsable
from distributions.lp.special import fast_log
from distributions.io.stream import json_stream_load, json_stream_dump
parsable = parsable.Parsable()

assert exp  # pacify pyflakes


DEFAULT_MAX_SIZE = 47

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.join(ROOT, 'clustering.data')
if not os.path.exists(TEMP):
    os.makedirs(TEMP)

CACHE = {}


def savefig(stem):
    for extension in ['png', 'pdf']:
        name = '{}/{}.{}'.format(TEMP, stem, extension)
        print 'saving', name
        pyplot.tight_layout()
        pyplot.savefig(name)


def get_larger_counts(small_counts):
    large_counts = defaultdict(lambda: 0.0)
    for small_shape, count in small_counts.iteritems():

        # create a new partition
        large_shape = (1,) + small_shape
        large_counts[large_shape] += count

        # add to each existing partition
        for i in xrange(len(small_shape)):
            large_shape = list(small_shape)
            large_shape[i] += 1
            large_shape.sort()
            large_shape = tuple(large_shape)
            large_counts[large_shape] += count

    return dict(large_counts)


def get_smaller_probs(small_counts, large_counts, large_probs):
    assert len(large_counts) == len(large_probs)
    small_probs = {}
    for small_shape, count in small_counts.iteritems():
        prob = 0.0

        # create a new partition
        large_shape = (1,) + small_shape
        prob += large_probs[large_shape] / large_counts[large_shape]

        # add to each existing partition
        for i in xrange(len(small_shape)):
            large_shape = list(small_shape)
            large_shape[i] += 1
            large_shape.sort()
            large_shape = tuple(large_shape)
            prob += large_probs[large_shape] / large_counts[large_shape]

        small_probs[small_shape] = count * prob
    return small_probs


def get_counts(size):
    '''
    Count partition shapes of a given sample size.

    Inputs:
        size = sample_size
    Returns:
        dict : shape -> count
    '''
    assert 0 <= size
    cache_file = '{}/counts.{}.json.bz2'.format(TEMP, size)
    if cache_file not in CACHE:
        if os.path.exists(cache_file):
            flat = json_stream_load(cache_file)
            large = {tuple(key): val for key, val in flat}
        else:
            if size == 0:
                large = {(): 1.0}
            else:
                small = get_counts(size - 1)
                large = get_larger_counts(small)
            print 'caching', cache_file
            json_stream_dump(large.iteritems(), cache_file)
        CACHE[cache_file] = large
    return CACHE[cache_file]


def enum_counts(max_size):
    return [get_counts(size) for size in range(1 + max_size)]


def get_log_z(shape):
    return sum(n * math.log(n) for n in shape)


def get_log_Z(counts):
    return numpy.logaddexp.reduce([
        get_log_z(shape) + math.log(count)
        for shape, count in counts.iteritems()
    ])


def get_probs(size):
    counts = get_counts(size).copy()
    for shape, count in counts.iteritems():
        counts[shape] = get_log_z(shape) + math.log(count)
    log_Z = numpy.logaddexp.reduce(counts.values())
    for shape, log_z in counts.iteritems():
        counts[shape] = math.exp(log_z - log_Z)
    return counts


def get_subprobs(size, max_size):
    '''
    Compute probabilities of shapes of partial assignment vectors.

    Inputs:
        size = sample_size
        max_size = dataset_size
    Returns:
        dict : shape -> prob
    '''
    assert 0 <= size
    assert size <= max_size
    cache_file = '{}/subprobs.{}.{}.json.bz2'.format(TEMP, size, max_size)
    if cache_file not in CACHE:
        if os.path.exists(cache_file):
            flat = json_stream_load(cache_file)
            small_probs = {tuple(key): val for key, val in flat}
        else:
            if size == max_size:
                small_probs = get_probs(size)
            else:
                small_counts = get_counts(size)
                large_counts = get_counts(size + 1)
                large_probs = get_subprobs(size + 1, max_size)
                small_probs = get_smaller_probs(
                    small_counts,
                    large_counts,
                    large_probs)
            print 'caching', cache_file
            json_stream_dump(small_probs.iteritems(), cache_file)
        CACHE[cache_file] = small_probs
    return CACHE[cache_file]


def enum_probs(max_size):
    return [get_probs(size) for size in range(max_size + 1)]


@parsable.command
def priors(N=100):
    '''
    Plots different partition priors.
    '''
    X = numpy.array(range(1, N + 1))

    def plot(Y, *args, **kwargs):
        Y = numpy.array(Y)
        Y -= numpy.logaddexp.reduce(Y)
        pyplot.plot(X, Y, *args, **kwargs)

    def crp(alpha):
        assert 0 < alpha
        prob = numpy.zeros(len(X))
        prob[1:] = log(X[1:] - 1)
        prob[0] = log(alpha)
        return prob

    def entropy():
        prob = numpy.zeros(len(X))
        n_log_n = lambda n: n * log(n)
        prob[1:] = n_log_n(X[1:]) - n_log_n(X[1:] - 1)
        return prob

    def plot_crp(alpha):
        plot(crp(eval(alpha)), label='CRP({})'.format(alpha))

    def plot_entropy():
        plot(entropy(), 'k--', linewidth=2, label='low-entropy')

    pyplot.figure(figsize=(8, 4))

    plot_entropy()
    plot_crp('0.01')
    plot_crp('0.1')
    plot_crp('exp(-1)')
    plot_crp('1.0')
    plot_crp('10.0')

    pyplot.title('Posterior Predictive Curves of Clustering Priors')
    pyplot.xlabel('category size')
    pyplot.ylabel('log(probability)')
    pyplot.xscale('log')
    pyplot.legend(loc='best')

    savefig('priors')


def get_pairwise(counts):
    size = sum(iter(counts).next())
    paired = 0.0
    for shape, prob in counts.iteritems():
        paired += prob * sum(n * (n - 1) for n in shape) / (size * (size - 1))
    return paired


@parsable.command
def pairwise(max_size=DEFAULT_MAX_SIZE):
    '''
    Plot probability that two points lie in same cluster,
    as function of data set size.
    '''
    all_counts = enum_probs(max_size)
    sizes = range(2, len(all_counts))
    probs = [get_pairwise(all_counts[i]) for i in sizes]

    pyplot.figure()
    pyplot.plot(sizes, probs, marker='.')
    pyplot.title('\n'.join([
        'Cohabitation probability depends on dataset size',
        '(unlike the CRP or PYP)'
    ]))
    pyplot.xlabel('# objects')
    pyplot.ylabel('P[two random objects in same cluster]')
    pyplot.xscale('log')
    # pyplot.yscale('log')
    pyplot.ylim(0, 1)
    savefig('pairwise')


def get_color_range(size):
    scale = 1.0 / (size - 1.0)
    return [
        (scale * t, 0.5, scale * (size - t - 1))
        for t in range(size)
    ]


def approximate_postpred_correction(subsample_size, dataset_size):
    t = numpy.log(1.0 * dataset_size / subsample_size)
    return t * (0.45 - 0.1 / subsample_size - 0.1 / dataset_size)


def ad_hoc_size_factor(subsample_size, dataset_size):
    return numpy.exp(
        approximate_postpred_correction(subsample_size, dataset_size))


@parsable.command
def postpred(subsample_size=10):
    '''
    Plot posterior predictive probability and approximations,
    fixing subsample size and varying cluster size and dataset size.
    '''
    size = subsample_size
    max_sizes = [size] + [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    max_sizes = sorted(set(s for s in max_sizes if s >= size))
    colors = get_color_range(len(max_sizes))
    pyplot.figure(figsize=(12, 8))
    Y_max = 0

    large_counts = get_counts(size)
    for max_size, color in zip(max_sizes, colors):
        large_probs = get_subprobs(size, max_size)
        small_probs = get_subprobs(size - 1, max_size)

        def plot(X, Y, **kwargs):
            pyplot.scatter(
                X, Y,
                color=color,
                edgecolors='none',
                **kwargs)

        plot([], [], label='max_size = {}'.format(max_size))

        max_small_prob = max(small_probs.itervalues())
        for small_shape, small_prob in small_probs.iteritems():
            X = []
            Y = []

            # create a new partition
            n = 1
            large_shape = (1,) + small_shape
            prob = large_probs[large_shape] / large_counts[large_shape]
            X.append(n)
            Y.append(prob)
            singleton_prob = prob

            # add to each existing partition
            for i in range(len(small_shape)):
                n = small_shape[i] + 1
                large_shape = list(small_shape)
                large_shape[i] += 1
                large_shape.sort()
                large_shape = tuple(large_shape)
                prob = large_probs[large_shape] / large_counts[large_shape]
                X.append(n)
                Y.append(prob)

            X = numpy.array(X)
            Y = numpy.array(Y)
            Y /= singleton_prob
            alpha = small_prob / max_small_prob
            plot(X, Y, alpha=alpha)
            Y_max = max(Y_max, max(Y))

    X = numpy.array(range(1, size + 1))

    # entropy
    entropy = numpy.array([
        x * (x / (x - 1.0)) ** (x - 1.0) if x > 1 else 1
        for x in X
    ])
    Y = entropy / entropy.min()
    pyplot.plot(X, Y, 'k--', label='entropy', linewidth=2)

    # CRP
    alpha = math.exp(-1)
    Y = numpy.array([x - 1 if x > 1 else alpha for x in X])
    Y /= Y.min()
    pyplot.plot(X, Y, 'g-', label='CRP(exp(-1))'.format(alpha))

    # ad hoc
    factors = ad_hoc_size_factor(size, numpy.array(max_sizes))
    for factor in factors:
        Y = entropy.copy()
        Y[0] *= factor
        Y /= Y.min()
        pyplot.plot(X, Y, 'r--')
    pyplot.plot([], [], 'r--', label='ad hoc')

    pyplot.yscale('log')
    pyplot.xscale('log')
    pyplot.title(
        'Adding 1 point to subsample of {} points out of {} total'.format(
            size, max_size))
    pyplot.xlabel('cluster size')
    pyplot.ylabel('posterior predictive probability')
    pyplot.xlim(1, size * 1.01)
    pyplot.ylim(1, Y_max * 1.01)
    pyplot.legend(
        prop=font_manager.FontProperties(size=10),
        loc='upper left')
    savefig('postpred')


def true_postpred_correction(subsample_size, dataset_size):
    '''
    Compute true postpred constant according to size-based approximation.
    '''
    large_counts = get_counts(subsample_size)
    large_probs = get_subprobs(subsample_size, dataset_size)
    small_probs = get_subprobs(subsample_size - 1, dataset_size)

    numer = 0
    denom = 0

    for small_shape, small_prob in small_probs.iteritems():
        probs = []

        # create a new partition
        n = 1
        large_shape = (1,) + small_shape
        prob = large_probs[large_shape] / large_counts[large_shape]
        probs.append((n, prob))

        # add to each existing partition
        for i in range(len(small_shape)):
            n = small_shape[i] + 1
            large_shape = list(small_shape)
            large_shape[i] += 1
            large_shape.sort()
            large_shape = tuple(large_shape)
            prob = large_probs[large_shape] / large_counts[large_shape]
            probs.append((n, prob))

        total = sum(prob for _, prob in probs)
        singleton_prob = probs[0][1]
        for n, prob in probs[1:]:
            weight = small_prob * prob / total
            baseline = -math.log(n * (n / (n - 1.0)) ** (n - 1.0))
            correction = math.log(singleton_prob / prob) - baseline
            numer += weight * correction
            denom += weight

    return numer / denom if denom > 0 else 1.0


@parsable.command
def dataprob(subsample_size=10, dataset_size=50):
    '''
    Plot data prob approximation.

    This tests the accuracy of LowEntropy.score_counts(...).
    '''
    true_probs = get_subprobs(subsample_size, dataset_size)
    naive_probs = get_probs(subsample_size)
    shapes = true_probs.keys()

    # apply ad hoc size factor
    approx_probs = naive_probs.copy()
    factor = ad_hoc_size_factor(subsample_size, dataset_size)
    print 'factor =', factor
    for shape in shapes:
        approx_probs[shape] *= factor ** (len(shape) - 2)

    X = numpy.array([true_probs[shape] for shape in shapes])
    Y0 = numpy.array([naive_probs[shape] for shape in shapes])
    Y1 = numpy.array([approx_probs[shape] for shape in shapes])

    pyplot.figure()
    pyplot.scatter(X, Y0, color='blue', edgecolors='none', label='naive')
    pyplot.scatter(X, Y1, color='red', edgecolors='none', label='approx')
    pyplot.xlabel('true probability')
    pyplot.ylabel('approximation')
    LB = min(X.min(), Y0.min(), Y1.min())
    UB = max(X.max(), Y0.max(), Y1.max())
    pyplot.xlim(LB, UB)
    pyplot.ylim(LB, UB)
    pyplot.plot([LB, UB], [LB, UB], 'k--')
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.title('\n'.join([
        'Approximate data probability',
        'subsample_size = {}, dataset_size = {}'.format(
            subsample_size,
            dataset_size),
    ]))
    pyplot.legend(
        prop=font_manager.FontProperties(size=10),
        loc='lower right')
    savefig('dataprob')


def true_dataprob_correction(subsample_size, dataset_size):
    '''
    Compute true normalization correction.
    '''
    naive_probs = get_probs(subsample_size)
    factor = ad_hoc_size_factor(subsample_size, dataset_size)
    Z = sum(
        prob * factor ** (len(shape) - 1)
        for shape, prob in naive_probs.iteritems()
    )
    return -math.log(Z)


def approximate_dataprob_correction(subsample_size, dataset_size):
    n = math.log(subsample_size)
    N = math.log(dataset_size)
    return 0.061 * n * (n - N) * (n + N) ** 0.75


@parsable.command
def normalization(max_size=DEFAULT_MAX_SIZE):
    '''
    Plot approximation to partition function of low-entropy clustering
    distribution for various set sizes.
    '''
    pyplot.figure()

    all_counts = enum_counts(max_size)
    sizes = numpy.array(range(1, 1 + max_size))
    log_Z = numpy.array([
        get_log_Z(all_counts[size]) for size in sizes
    ])
    log_z_max = numpy.array([get_log_z([size]) for size in sizes])

    coeffs = [
        (log_Z[i] / log_z_max[i] - 1.0) * sizes[i] ** 0.75
        for i in [1, -1]
    ]
    coeffs += [0.27, 0.275, 0.28]
    for coeff in coeffs:
        print coeff
        approx = log_z_max * (1 + coeff * sizes ** -0.75)
        X = sizes ** -0.75
        Y = (log_Z - approx) / log_Z
        pyplot.plot(X, Y, marker='.', label='coeff = {}'.format(coeff))
    pyplot.xlim(0, 1)
    pyplot.xlabel('1 / size')
    pyplot.ylabel('approx error')
    pyplot.title(
        'log(Z) ~ log(z_max) * (1 + coeff * size ** -0.75)')
    pyplot.legend(loc='best')
    savefig('normalization')


@parsable.command
def approximations(max_size=DEFAULT_MAX_SIZE):
    '''
    Plot both main approximations for many (subsample, dataset) sizes:
    (1) normalization constant, and
    (2) postpred size factor
    '''
    sizes = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60]
    sizes = [size for size in sizes if size <= max_size]
    keys = [(x, y) for x in sizes for y in sizes if x <= y]

    truth1 = {}
    truth2 = {}
    approx1 = {}
    approx2 = {}
    for key in keys:
        size, max_size = key

        # postpred correction
        if size > 1:
            truth1[key] = true_postpred_correction(size, max_size)
            approx1[key] = approximate_postpred_correction(size, max_size)

        # normalization correction
        truth2[key] = true_dataprob_correction(size, max_size)
        approx2[key] = approximate_dataprob_correction(size, max_size)

    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.set_title('Approximation accuracies of postpred and dataprob')
    ax2.set_ylabel('log(Z correction)')
    ax1.set_ylabel('log(singleton postpred correction)')
    ax2.set_xlabel('subsample size')
    ax1.set_xlim(min(sizes) * 0.95, max(sizes) * 1.05)
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    def plot(ax, X, y, values, *args, **kwargs):
        Y = [values[x, y] for x in X if (x, y) in values]
        X = [x for x in X if (x, y) in values]
        ax.plot(X, Y, *args, alpha=0.5, marker='.', **kwargs)

    for max_size in sizes:
        X = [n for n in sizes if n <= max_size]
        plot(ax1, X, max_size, truth1, 'k-')
        plot(ax1, X, max_size, approx1, 'r-')
        plot(ax2, X, max_size, truth2, 'k-')
        plot(ax2, X, max_size, approx2, 'r-')

    plot(ax1, [], None, {}, 'r-', label='approximation')
    plot(ax1, [], None, {}, 'k-', label='truth')
    ax1.legend(loc='upper right')

    savefig('approximations')


@parsable.command
def fastlog():
    '''
    Plot accuracy of fastlog term in cluster_add_score.
    '''
    X = numpy.array([2.0 ** i for i in range(20 + 1)])
    Y0 = numpy.array([x * math.log(1. + 1. / x) for x in X])
    Y1 = numpy.array([x * fast_log(1. + 1. / x) for x in X])
    Y2 = numpy.array([1.0 for x in X])

    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True)

    ax1.plot(X, Y0, 'ko', label='math.log')
    ax1.plot(X, Y1, 'r-', label='lp.special.fast_log')
    ax1.plot(X, Y2, 'b-', label='asymptote')
    ax2.plot(X, numpy.abs(Y1 - Y0), 'r-', label='lp.special.fast_log')
    ax2.plot(X, numpy.abs(Y2 - Y0), 'b-', label='asymptote')

    ax1.set_title('lp.special.fast_log approximation')
    ax1.set_ylabel('n log(1 + 1 / n)')
    ax2.set_ylabel('approximation error')
    ax2.set_xlabel('n')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    savefig('fastlog')


def number_table(numbers, width=5):
    lines = []
    line = ''
    for i, number in enumerate(numbers):
        if i % width == 0:
            if line:
                lines.append(line + ',')
            line = '    %0.8f' % number
        else:
            line += ', %0.8f' % number
    if line:
        lines.append(line)
    return '\n'.join(lines)


@parsable.command
def code(max_size=DEFAULT_MAX_SIZE):
    '''
    Generate C++ code for clustering partition function.
    '''
    all_counts = enum_counts(max_size)
    sizes = range(1 + max_size)
    log_Z = [
        get_log_Z(all_counts[size]) if size else 0
        for size in sizes
    ]
    size = sizes[-1]
    coeff = (log_Z[-1] / get_log_z([size]) - 1.0) * size ** 0.75

    print '# Insert this in src/clustering.cc:'
    lines = [
        '// this code was generated by derivations/clustering.py',
        'static const float log_partition_function_table[%d] =' %
        (max_size + 1),
        '{',
        number_table(log_Z),
        '};',
        '',
        '// this code was generated by derivations/clustering.py',
        'template<class count_t>',
        'float Clustering<count_t>::LowEntropy::log_partition_function (',
        '        count_t sample_size) const',
        '{',
        '    // TODO incorporate dataset_size for higher accuracy',
        '    count_t n = sample_size;',
        '    if (n < %d) {' % (max_size + 1),
        '        return log_partition_function_table[n];',
        '    } else {',
        '        float coeff = %0.8ff;' % coeff,
        '        float log_z_max = n * fast_log(n);',
        '        return log_z_max * (1.f + coeff * powf(n, -0.75f));',
        '    }',
        '}',
    ]
    print '\n'.join(lines)
    print

    print '# Insert this in distributions/dbg/clustering.py:'
    lines = [
        '# this code was generated by derivations/clustering.py',
        'log_partition_function_table = [',
        number_table(log_Z),
        ']',
        '',
        '',
        '# this code was generated by derivations/clustering.py',
        'def log_partition_function(sample_size):',
        '    # TODO incorporate dataset_size for higher accuracy',
        '    n = sample_size',
        '    if n < %d:' % (max_size + 1),
        '        return LowEntropy.log_partition_function_table[n]',
        '    else:',
        '        coeff = %0.8f' % coeff,
        '        log_z_max = n * log(n)',
        '        return log_z_max * (1.0 + coeff * n ** -0.75)',
    ]
    print '\n'.join(lines)
    print


@parsable.command
def plots():
    '''
    Generate all plots.
    '''
    priors()
    pairwise()
    postpred()
    dataprob()
    approximations()
    normalization()
    fastlog()


if __name__ == '__main__':
    parsable.dispatch()
