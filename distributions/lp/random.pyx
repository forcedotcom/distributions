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
