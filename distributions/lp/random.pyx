from libc.math cimport exp
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from python cimport PyObject
from distributions.lp.random cimport rng_t, global_rng

ctypedef PyObject* O


cdef extern from "distributions/random.hpp" namespace "distributions":
    cdef pair[size_t, float] sample_prob_from_scores_overwrite (
            rng_t & rng,
            vector[float] & scores)
    cdef float score_from_scores_overwrite (
            rng_t & rng,
            size_t sample,
            vector[float] & scores)
    cdef pair[O, O] _sample_pair_from_urn \
            "distributions::sample_pair_from_urn<PyObject *>" (
            rng_t & rng,
            vector[O] & urn) nogil


cdef class RNG:
    cdef rng_t * ptr
    def __cinit__(self):
        self.ptr = new rng_t()
    def __dealloc__(self):
        del self.ptr
    def seed(self, int n):
        self.ptr.seed(n)
    def __call__(self):
        return self.ptr.sample()
    def copy(self):
        cdef RNG result = RNG()
        del result.ptr
        result.ptr = new rng_t(self.ptr[0])
        return result


def sample_prob_from_scores(RNG rng, list scores):
    cdef vector[float] _scores = scores
    return sample_prob_from_scores_overwrite(rng.ptr[0], _scores)


def prob_from_scores(RNG rng, int sample, list scores):
    cdef vector[float] _scores = scores
    cdef float score = score_from_scores_overwrite(rng.ptr[0], sample, _scores)
    cdef float prob = exp(score)
    return prob


def sample_pair_from_urn(list urn):
    cdef vector[O] _urn
    for item in urn:
        _urn.push_back(<O> item)
    cdef pair[O, O] result = _sample_pair_from_urn(global_rng, _urn)
    return (<object> result.first, <object> result.second)
