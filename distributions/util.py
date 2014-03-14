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

import numpy
import scipy.stats
from collections import defaultdict


def scores_to_probs(scores):
    scores = numpy.array(scores)
    scores -= scores.max()
    probs = numpy.exp(scores)
    probs /= probs.sum()
    return probs


def score_to_empirical_kl(score, count):
    """
    Convert total log score to KL( empirical || model ),
    where the empirical pdf is uniform over `count` datapoints.
    """
    count = float(count)
    return -score / count - numpy.log(count)


def print_histogram(probs, counts):
    WIDTH = 60.0
    max_count = max(counts)
    print '{: >8} {: >8}'.format('Prob', 'Count')
    for prob, count in sorted(zip(probs, counts), reverse=True):
        width = int(round(WIDTH * count / max_count))
        print '{: >8.3f} {: >8d} {}'.format(prob, count, '-' * width)


def multinomial_goodness_of_fit(
        probs,
        counts,
        total_count,
        truncated=False,
        plot=False):
    """
    Pearson's chi^2 test, on possibly truncated data.
    http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

    Returns:
        p-value of truncated multinomial sample.
    """
    assert len(probs) == len(counts)
    assert truncated or total_count == sum(counts)
    chi_squared = 0
    dof = 0
    if plot:
        print_histogram(probs, counts)
    for p, c in zip(probs, counts):
        if p == 1:
            return 1 if c == total_count else 0
        assert p < 1, 'bad probability: %g' % p
        if p > 0:
            mean = total_count * p
            variance = total_count * p * (1 - p)
            assert variance > 1,\
                'WARNING goodness of fit is inaccurate; use more samples'
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


def unif01_goodness_of_fit(samples, plot=False):
    """
    Bin uniformly distributed samples and apply Pearson's chi^2 test.
    """
    samples = numpy.array(samples, dtype=float)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0
    bin_count = int(round(len(samples) ** 0.333))
    assert bin_count >= 7, 'WARNING imprecise test, use more samples'
    probs = numpy.ones(bin_count, dtype=numpy.float) / bin_count
    counts = numpy.zeros(bin_count, dtype=numpy.int)
    for sample in samples:
        counts[int(bin_count * sample)] += 1
    return multinomial_goodness_of_fit(probs, counts, len(samples), plot=plot)


def density_goodness_of_fit(samples, probs, plot=False):
    """
    Transform arbitrary continuous samples to unif01 distribution
    and assess goodness of fit via Pearson's chi^2 test.

    Inputs:
        samples - a list of real-valued samples from a distribution
        probs - a list of probability densities evaluated at those samples
    """
    assert len(samples) == len(probs)
    assert len(samples) > 100, 'WARNING imprecision; use more samples'
    pairs = zip(samples, probs)
    pairs.sort()
    samples = numpy.array([x for x, p in pairs])
    probs = numpy.array([p for x, p in pairs])
    density = numpy.sqrt(probs[1:] * probs[:-1])
    gaps = samples[1:] - samples[:-1]
    unif01_samples = 1.0 - numpy.exp(-len(samples) * gaps * density)
    return unif01_goodness_of_fit(unif01_samples, plot=plot)


def discrete_goodness_of_fit(
        samples,
        probs_dict,
        truncate_beyond=8,
        plot=False):
    """
    Transform arbitrary discrete data to multinomial
    and assess goodness of fit via Pearson's chi^2 test.
    """
    assert len(samples) > 100, 'WARNING imprecision; use more samples'
    counts = defaultdict(lambda: 0)
    for sample in samples:
        assert sample in probs_dict
        counts[sample] += 1
    items = [(prob, counts.get(i, 0)) for i, prob in probs_dict.iteritems()]
    items.sort(reverse=True)
    truncated = (truncate_beyond and truncate_beyond < len(items))
    if truncated:
        items = items[:truncate_beyond]
    probs = [prob for prob, count in items]
    counts = [count for prob, count in items]
    return multinomial_goodness_of_fit(
        probs,
        counts,
        len(samples),
        truncated=truncated,
        plot=plot)


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
    http://stackoverflow.com/questions/2659900
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
    if bin_count is None:
        bin_count = numpy.max(samples) + 1
    v = numpy.zeros(bin_count, dtype=int)
    for sample in samples:
        v[sample] += 1
    return v
