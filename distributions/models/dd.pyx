from libc.math cimport log
from distributions.scimath cimport gammaln, dirichlet_draw, categorical_draw
import numpy
cimport numpy

cpdef int MAX_DIM = 256

cdef extern from "common.hpp" namespace "distributions":
    cppclass rng_t:
        pass


cdef class model_t:
    cdef double[256] alphas
    cdef int dim
    def __init__(self, alphas):
        cdef int dim = len(alphas)
        assert dim <= 256
        self.dim = dim
        for i in xrange(dim):
            self.alphas[i] = alphas[i]


cdef class group_t:
    cdef int[256] counts
    def __init__(self, counts):
        cdef int dim = len(counts)
        assert dim <= 256
        for i in xrange(dim):
            self.counts[i] = counts[i]


def model_load(model):
    return model_t(numpy.array(model['alphas'], dtype=numpy.float32))


def model_dump(model_t model):
    return {'alphas': [model.alphas[i] for i in xrange(model.dim)]}


def group_load(group=None, model=None):
    return group_t(numpy.array(group['counts'], dtype=numpy.int32))


def group_dump(group_t group):
    return {'counts': [group.counts[i] for i in xrange(group.dim)]}


def add_data(group_t group, unsigned y):
    group.counts[y] += 1


def remove_data(group_t group, unsigned y):
    group.counts[y] -= 1


def sample_data(model_t model, group_t group):
    cdef double[256] ps
    _sample_post(model, group, ps)
    return categorical_draw(model.D, ps)


cdef _sample_post(model_t model, group_t group, double *thetas):
    cdef double[256] alpha_n
    cdef int i
    for i in xrange(model.D):
        alpha_n[i] = group.counts[i] + model.alphas[i]
    dirichlet_draw(model.D, alpha_n, thetas)


cpdef sample_post(group_t group, model_t model):
    cdef double[256] thetas
    _sample_post(model, group, thetas)
    r = numpy.zeros(model.D)
    cdef int i
    for i in xrange(model.D):
        r[i] = thetas[i]
    return r


def generate_post(group_t group, model_t model):
    post = sample_post(group, model)
    return {'p': post.tolist()}


cpdef double pred_prob(unsigned y, group_t group, model_t model):
    """
    McCallum, et. al, 'Rething LDA: Why Priors Matter' eqn 4
    """
    cdef double sum = 0.
    cdef int i
    for i in xrange(model.D):
        sum += group.counts[i] + model.alphas[i]
    return log((group.counts[y] + model.alphas[y]) / sum)


def score_group(group_t group, model_t model):
    """
    From equation 22 of Michael Jordan's CS281B/Stat241B
    Advanced Topics in Learning and Decision Making course,
    'More on Marginal Likelihood'
    """
    cdef int i
    cdef double alpha_sum = 0.
    cdef int count_sum = 0
    cdef double sum = 0.
    for i in xrange(model.D):
        alpha_sum += model.alphas[i]
    for i in xrange(model.D):
        count_sum += group.counts[i]
    for i in xrange(model.D):
        sum += gammaln(model.alphas[i] + group.counts[i]) - gammaln(model.alphas[i])
    return sum + gammaln(alpha_sum) - gammaln(alpha_sum + count_sum)


cpdef add_pred_probs(
        model_t model,
        list groups,
        unsigned y,
        numpy.ndarray[double, ndim=1] scores):
    """
    Vectorize over i: scores[i] += pred_prob(model[i], groups[i], y)
    """
    cdef int size = len(scores)
    assert len(groups) == size
    cdef int i
    for i in xrange(size):
        scores[i] += pred_prob(model, groups[i], y)


cdef extern from "models/dd.hpp" namespace "distributions":
    int foo()


def wrapped_foo():
    return foo() + 1
