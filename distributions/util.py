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


def scores_to_probs(scores):
    scores = numpy.array(scores)
    scores -= scores.max()
    probs = numpy.exp(scores, out=scores)
    probs /= probs.sum()
    return probs


def score_to_empirical_kl(score, count):
    """
    Convert total log score to KL( empirical || model ),
    where the empirical pdf is uniform over `count` datapoints.
    """
    count = float(count)
    return -score / count - numpy.log(count)


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
    # We need to distribute the remainder relatively evenly;
    # tests will be inaccurate if we have small bins at the end.
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
