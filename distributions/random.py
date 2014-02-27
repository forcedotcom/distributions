import numpy.random
from numpy.random.mtrand import dirichlet as sample_dirichlet


def sample_discrete(ps):
    """
    Draws from a discrete distribution with the given (possibly unnormalized)
    probabilities for each outcome.

    Returns an int between 0 and len(ps)-1, inclusive
    """
    z = float(sum(ps))
    a = numpy.random.rand()
    tot = 0.0
    for i in range(len(ps)):
        tot += ps[i] / z
        if a < tot:
            return i
    raise ValueError('bug in sample_discrete')


def seed(x):
    numpy.random.seed(x)
    try:
        import distributions.cRandom
        distributions.cRandom.seed(x)
    except ImportError:
        pass
