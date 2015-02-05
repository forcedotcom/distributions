from itertools import izip
import numpy
from numpy import pi
from scipy.special import gamma
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors
import parsable
from distributions.dbg.models import niw
from nose.tools import assert_almost_equal


def get_edge_stats(module, EXAMPLE, sample_count):
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    group = module.Group.from_values(shared, values)
    sampler = module.Sampler()
    sampler.init(shared, group)

    samples = []
    scores = []
    for _ in xrange(sample_count):
        value = sampler.eval(shared)
        samples.append(value)
        score = group.score_value(shared, value)
        scores.append(score)

    neighbors = NearestNeighbors(n_neighbors=2).fit(samples)
    distances, indices = neighbors.kneighbors(samples)
    edge_lengths = distances[:, 1]
    nearest = indices[:, 1]
    edge_scores = numpy.array([
        0.5 * (scores[i] + scores[j])
        for i, j in enumerate(nearest)
    ])

    return {'lengths': edge_lengths, 'scores': edge_scores}


@parsable.command
def plot_edges(sample_count=1000):
    '''
    Plot edges of niw examples.
    '''
    fig, axes = pyplot.subplots(
        len(niw.EXAMPLES),
        2,
        sharey='row',
        figsize=(8, 12))

    for EXAMPLE, (ax1, ax2) in izip(niw.EXAMPLES, axes):
        dim = len(EXAMPLE['shared']['mu'])
        edges = get_edge_stats(niw, EXAMPLE, sample_count)

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


def volume_of_sphere(dim, radius=1.0):
    return radius ** dim * pi ** (0.5 * dim) / gamma(0.5 * dim + 1)


def test_volume_of_sphere():
    for r in [0.1, 1.0, 10.0]:
        assert_almost_equal(volume_of_sphere(1, r), 2.0 * r)
        assert_almost_equal(volume_of_sphere(2, r), pi * r ** 2)
        assert_almost_equal(volume_of_sphere(3, r), 4/3.0 * pi * r ** 3)


@parsable.command
def plot_cdf(sample_count=1000):
    '''
    Plot test statistic cdf based on the Nearest Neighbor distribution [1].

    [1] http://en.wikipedia.org/wiki/Nearest_neighbour_distribution
    [2] http://en.wikipedia.org/wiki/Volume_of_an_n-ball
    '''
    pyplot.figure()

    for EXAMPLE in niw.EXAMPLES:
        dim = len(EXAMPLE['shared']['mu'])
        edges = get_edge_stats(niw, EXAMPLE, sample_count)
        radii = edges['lengths']
        intensities = sample_count * numpy.exp(edges['scores'])

        cdf = [
            1 - numpy.exp(-intensity * volume_of_sphere(dim, radius))
            for intensity, radius in izip(intensities, radii)
        ]
        cdf.sort()

        X = numpy.arange(0.5 / sample_count, 1, 1.0 / sample_count)

        pyplot.plot(X, cdf, label='dim = {}'.format(dim))

    pyplot.title('Nearest Neighbor Distance CDF')
    pyplot.legend(loc='best')
    pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    parsable.dispatch()
