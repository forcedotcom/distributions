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

from libc.math cimport exp
from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport numpy
numpy.import_array()
from cpython cimport PyObject
from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng
from distributions._eigen_h cimport VectorXf, MatrixXf
from distributions._eigen cimport to_eigen_vecf, to_eigen_matf

ctypedef PyObject* O


cdef extern from "distributions/random.hpp" namespace "distributions":
    cdef float log_sum_exp_cc "distributions::log_sum_exp" (
            vector[float] & scores) nogil
    cdef pair[size_t, float] sample_prob_from_scores_overwrite (
            rng_t & rng,
            vector[float] & scores) nogil
    cdef float score_from_scores_overwrite (
            rng_t & rng,
            size_t sample,
            vector[float] & scores) nogil
    cdef pair[O, O] sample_pair_from_urn_cc \
            "distributions::sample_pair_from_urn<PyObject *>" (
            rng_t & rng,
            vector[O] & urn) nogil
    cdef size_t sample_discrete_cc "distributions::sample_discrete" (
            rng_t & rng,
            size_t dim,
            const float * probs) nogil
    cdef float score_student_t_cc "distributions::score_mv_student_t" (
            const VectorXf &,
            float,
            const VectorXf &,
            const MatrixXf &) nogil

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


def log_sum_exp(list scores):
    cdef vector[float] _scores = scores
    return log_sum_exp_cc(_scores)


def sample_prob_from_scores(list scores):
    cdef vector[float] _scores = scores
    return sample_prob_from_scores_overwrite(get_rng()[0], _scores)


def prob_from_scores(int sample, list scores):
    cdef vector[float] _scores = scores
    cdef float score = score_from_scores_overwrite(get_rng()[0], sample, _scores)
    cdef float prob = exp(score)
    return prob


def sample_pair_from_urn(list urn):
    cdef vector[O] _urn
    for item in urn:
        _urn.push_back(<O> item)
    cdef pair[O, O] result = sample_pair_from_urn_cc(get_rng()[0], _urn)
    return (<object> result.first, <object> result.second)


def sample_discrete(numpy.ndarray[numpy.float32_t, ndim=1] probs):
    cdef size_t size = probs.shape[0]
    cdef float * data = <float *> probs.data
    return sample_discrete_cc(get_rng()[0], size, data)

def score_student_t(numpy.ndarray v,
                    float nu,
                    numpy.ndarray mu,
                    numpy.ndarray sigma):
    cdef VectorXf c_v = to_eigen_vecf(v)
    cdef VectorXf c_mu = to_eigen_vecf(mu)
    cdef MatrixXf c_sigma = to_eigen_matf(sigma)
    return score_student_t_cc(c_v, nu, c_mu, c_sigma)
