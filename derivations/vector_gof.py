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

from itertools import izip
import numpy
from scipy.stats import norm
from scipy.stats import multivariate_normal  # requires version >=0.14
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors
import parsable
from distributions.dbg.models import nich
from distributions.dbg.models import niw
from distributions.lp.models import nich as lp_nich
from distributions.lp.models import niw as lp_niw
from distributions.util import volume_of_sphere
from distributions.tests.util import seed_all


def get_dim(value):
    if isinstance(value, float):
        return 1
    else:
        return len(value)


def get_samples(model, EXAMPLE, sample_count):
    shared = model.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    group = model.Group.from_values(shared, values)

    # This version seems to be broken
    # sampler = model.Sampler()
    # sampler.init(shared, group)
    # ...
    # for _ in xrange(sample_count):
    #     value = sampler.eval(shared)

    samples = []
    scores = []
    for _ in xrange(sample_count):
        value = group.sample_value(shared)
        samples.append(value)
        score = group.score_value(shared, value)
        scores.append(score)

    return numpy.array(samples), numpy.array(scores)


def get_edge_stats(samples, scores):
    if not hasattr(samples[0], '__iter__'):
        samples = numpy.array([samples]).T
    neighbors = NearestNeighbors(n_neighbors=2).fit(samples)
    distances, indices = neighbors.kneighbors(samples)
    return {'lengths': distances[:, 1], 'scores': scores}


@parsable.command
def plot_edges(sample_count=1000, seed=0):
    '''
    Plot edges of niw examples.
    '''
    seed_all(seed)
    fig, axes = pyplot.subplots(
        len(niw.EXAMPLES),
        2,
        sharey='row',
        figsize=(8, 12))

    model = niw
    for EXAMPLE, (ax1, ax2) in izip(model.EXAMPLES, axes):
        dim = get_dim(EXAMPLE['shared']['mu'])
        samples, scores = get_samples(model, EXAMPLE, sample_count)
        edges = get_edge_stats(samples, scores)

        edge_lengths = numpy.log(edges['lengths'])
        edge_scores = edges['scores']
        edge_stats = [
            numpy.exp((s - d) / dim)
            for d, s in izip(edge_lengths, edge_scores)
        ]

        ax1.set_title('NIW, dim = {}'.format(dim))
        ax1.scatter(edge_lengths, edge_scores, lw=0, alpha=0.5)
        ax1.set_ylabel('log(edge prob)')

        ax2.scatter(edge_stats, edge_scores, lw=0, alpha=0.5)
        ax2.yaxis.set_label_position('right')

    ax1.set_xlabel('log(edge length)')
    ax2.set_ylabel('statistic')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    pyplot.show()


def cdf_to_pdf(Y, X, bandwidth=0.1):
    assert len(Y) == len(X)
    shift = max(1, int(round(len(Y) * bandwidth)))
    Y = (1.0 / shift) * (Y[shift:] - Y[:-shift])
    X = 0.5 * (X[shift:] + X[:-shift])
    return Y, X


@parsable.command
def plot_cdf(sample_count=1000, seed=0):
    '''
    Plot test statistic cdf based on the Nearest Neighbor distribution [1,2,3].

    [1] http://projecteuclid.org/download/pdf_1/euclid.aop/1176993668
    [2] http://arxiv.org/pdf/1006.3019v2.pdf
    [3] http://en.wikipedia.org/wiki/Nearest_neighbour_distribution
    [4] http://en.wikipedia.org/wiki/Volume_of_an_n-ball
    '''
    seed_all(seed)

    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True, figsize=(8, 10))
    ax1.plot([0, 1], [0, 1], 'k--')
    ax2.plot([0, 1], [1, 1], 'k--')

    for model in [nich, lp_nich, niw, lp_niw]:
        name = model.__name__.replace('distributions.', '')
        name = name.replace('models.', '')
        for EXAMPLE in model.EXAMPLES:
            dim = get_dim(EXAMPLE['shared']['mu'])
            samples, scores = get_samples(model, EXAMPLE, sample_count)
            edges = get_edge_stats(samples, scores)
            radii = edges['lengths']
            intensities = sample_count * numpy.exp(edges['scores'])

            cdf = numpy.array([
                1 - numpy.exp(-intensity * volume_of_sphere(dim, radius))
                for intensity, radius in izip(intensities, radii)
            ])
            cdf.sort()
            X = numpy.arange(0.5 / sample_count, 1, 1.0 / sample_count)

            pdf, Xp = cdf_to_pdf(cdf, X)
            pdf *= sample_count

            error = 2 * (sum(cdf) / sample_count) - 1
            if abs(error) < 0.05:
                status = 'PASS'
                linestyle = '-'
            else:
                status = 'FAIL'
                linestyle = '--'
            label = '{} {}({}) error = {:.3g}'.format(status, name, dim, error)
            ax1.plot(X, cdf, linestyle=linestyle, label=label)
            ax2.plot(Xp, pdf, linestyle=linestyle, label=label)

    ax1.set_title('GOF of Nearest Neighbor Statistic')
    ax1.legend(loc='best', prop={'size': 10}, fancybox=True, framealpha=0.5)
    ax1.set_ylabel('CDF')
    ax2.set_ylabel('PDF')
    pyplot.tight_layout()
    fig.subplots_adjust(hspace=0)
    pyplot.show()


def get_normal_example(sample_count):
    loc = 1.0
    scale = 2.0
    samples0 = norm.rvs(loc, scale, sample_count)
    samples1 = norm.rvs(loc, scale, sample_count)
    scores0 = norm.logpdf(samples0, loc, scale)
    scores1 = norm.logpdf(samples1, loc, scale)
    samples = numpy.array(zip(samples0, samples1))
    scores = scores0 + scores1
    return {'name': 'normal', 'samples': samples, 'scores': scores}


def get_mvn_example(sample_count):
    mean = numpy.array([1.0, 2.0])
    cov = numpy.array([[3.0, 2.0], [2.0, 3.0]])
    samples = multivariate_normal.rvs(mean, cov, sample_count)
    scores = multivariate_normal.logpdf(samples, mean, cov)
    return {'name': 'MVN', 'samples': samples, 'scores': scores}


def get_dbg_nich_example(sample_count):
    import distributions.lp.models.nich as model
    EXAMPLE = model.EXAMPLES[0]
    samples0, scores0 = get_samples(model, EXAMPLE, sample_count)
    samples1, scores1 = get_samples(model, EXAMPLE, sample_count)
    samples = numpy.array(zip(samples0, samples1))
    scores = scores0 + scores1
    return {'name': 'dbg.nich', 'samples': samples, 'scores': scores}


def get_lp_nich_example(sample_count):
    import distributions.lp.models.nich as model
    EXAMPLE = model.EXAMPLES[0]
    samples0, scores0 = get_samples(model, EXAMPLE, sample_count)
    samples1, scores1 = get_samples(model, EXAMPLE, sample_count)
    samples = numpy.array(zip(samples0, samples1))
    scores = scores0 + scores1
    return {'name': 'lp.nich', 'samples': samples, 'scores': scores}


def get_dbg_niw_example(sample_count):
    import distributions.dbg.models.niw as model
    for EXAMPLE in model.EXAMPLES:
        if get_dim(EXAMPLE['shared']['mu']) == 2:
            break
    samples, scores = get_samples(model, EXAMPLE, sample_count)
    return {'name': 'dbg.niw', 'samples': samples, 'scores': scores}


def get_lp_niw_example(sample_count):
    import distributions.lp.models.niw as model
    for EXAMPLE in model.EXAMPLES:
        if get_dim(EXAMPLE['shared']['mu']) == 2:
            break
    samples, scores = get_samples(model, EXAMPLE, sample_count)
    return {'name': 'lp.niw', 'samples': samples, 'scores': scores}


@parsable.command
def scatter(sample_count=1000, seed=0):
    '''
    Plot test statistic cdf for all datatpoints in a 2d dataset.
    '''
    seed_all(seed)

    examples = {
        (0, 0): get_normal_example,
        (1, 0): get_mvn_example,
        (0, 1): get_dbg_nich_example,
        (1, 1): get_lp_nich_example,
        (0, 2): get_dbg_niw_example,
        (1, 2): get_lp_niw_example,
    }

    rows = 1 + max(key[0] for key in examples)
    cols = 1 + max(key[1] for key in examples)
    fig, axes = pyplot.subplots(rows, cols, figsize=(12, 8))
    cmap = pyplot.get_cmap('bwr')

    for (row, col), get_example in examples.iteritems():
        example = get_example(sample_count)
        edges = get_edge_stats(example['samples'], example['scores'])
        radii = edges['lengths']
        intensities = sample_count * numpy.exp(edges['scores'])

        dim = 2
        cdf = numpy.array([
            1 - numpy.exp(-intensity * volume_of_sphere(dim, radius))
            for intensity, radius in izip(intensities, radii)
        ])
        error = 2 * (sum(cdf) / sample_count) - 1

        X = [value[0] for value in example['samples']]
        Y = [value[1] for value in example['samples']]
        colors = cdf

        ax = axes[row][col]
        ax.set_title('{} error = {:0.3g}'.format(example['name'], error))
        ax.scatter(X, Y, 50, alpha=0.5, c=colors, cmap=cmap)

    pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    parsable.dispatch()
