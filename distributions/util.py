import scipy
import scipy.stats
import logging
import numpy


LOG = logging.getLogger(__name__)


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


def multinomial_goodness_of_fit(probs, counts, total_count, truncated=False):
    """
    Returns p-value of truncated multinomial sample,
    using a pearson chi^2 test
    http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    """
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
