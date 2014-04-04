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

from libcpp.vector cimport vector
cimport numpy
numpy.import_array()
from cython.operator cimport dereference as deref, preincrement as inc
from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng
from distributions.lp.vector cimport (
    VectorFloat,
    vector_float_from_ndarray,
    vector_float_to_ndarray,
)
from distributions.sparse_counter cimport SparseCounter
from distributions.mixins import ComponentModel, Serializable


ctypedef unsigned Value


cdef extern from "distributions/models/dpd.hpp" namespace "distributions":
    cdef cppclass Model_cc "distributions::DirichletProcessDiscrete":
        float gamma
        float alpha
        float beta0
        vector[float] betas
        #cppclass Value
        cppclass Group:
            SparseCounter counts
            void init (Model_cc &, rng_t &) nogil except +
            void add_value (Model_cc &, Value &, rng_t &) nogil except +
            void remove_value (Model_cc &, Value &, rng_t &) nogil except +
            void merge (Model_cc &, Group &, rng_t &) nogil except +
        cppclass Sampler:
            vector[float] probs
        cppclass Scorer:
            vector[float] scores
        cppclass Classifier:
            vector[Group] groups
            vector[VectorFloat] scores
            VectorFloat scores_shift
            void init (Model_cc &, rng_t &) nogil except +
            void add_group (Model_cc &, rng_t &) nogil except +
            void remove_group (Model_cc &, size_t) nogil except +
            void add_value \
                (Model_cc &, size_t, Value &, rng_t &) nogil except +
            void remove_value \
                (Model_cc &, size_t, Value &, rng_t &) nogil except +
            void score_value \
                (Model_cc &, Value &, VectorFloat &, rng_t &) nogil except +

        void sampler_init (Sampler &, Group &, rng_t &) nogil except +
        Value sampler_eval (Sampler &, rng_t &) nogil except +
        Value sample_value (Group &, rng_t &) nogil except +
        float score_value (Group &, Value &, rng_t &) nogil except +
        float score_group (Group &, rng_t &) nogil except +


cdef class Group:
    cdef Model_cc.Group * ptr
    def __cinit__(self):
        self.ptr = new Model_cc.Group()
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        cdef SparseCounter * counts = & self.ptr.counts
        counts.clear()
        cdef dict raw_counts = raw['counts']
        cdef str i
        cdef int count
        for i, count in raw_counts.iteritems():
            counts.init_count(int(i), count)

    def dump(self):
        cdef dict counts = {}
        cdef SparseCounter.iterator it = self.ptr.counts.begin()
        cdef SparseCounter.iterator end = self.ptr.counts.end()
        while it != end:
            counts[str(deref(it).first)] = deref(it).second
            inc(it)
        return {'counts': counts}

    def init(self, Model_cy model):
        self.ptr.init(model.ptr[0], get_rng()[0])

    def add_value(self, Model_cy model, Value value):
        self.ptr.add_value(model.ptr[0], value, get_rng()[0])

    def remove_value(self, Model_cy model, Value value):
        self.ptr.remove_value(model.ptr[0], value, get_rng()[0])

    def merge(self, Model_cy model, Group source):
        self.ptr.merge(model.ptr[0], source.ptr[0], get_rng()[0])


cdef class Classifier:
    cdef Model_cc.Classifier * ptr
    def __cinit__(self):
        self.ptr = new Model_cc.Classifier()
    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.groups.size()

    def append(self, Group group):
        self.ptr.groups.push_back(group.ptr[0])

    def clear(self):
        self.ptr.groups.clear()

    def init(self, Model_cy model):
        self.ptr.init(model.ptr[0], get_rng()[0])

    def add_group(self, Model_cy model):
        self.ptr.add_group(model.ptr[0], get_rng()[0])

    def remove_group(self, Model_cy model, int groupid):
        self.ptr.remove_group(model.ptr[0], groupid)

    def add_value(self, Model_cy model, int groupid, Value value):
        self.ptr.add_value(model.ptr[0], groupid, value, get_rng()[0])

    def remove_value(self, Model_cy model, int groupid, Value value):
        self.ptr.remove_value(model.ptr[0], groupid, value, get_rng()[0])

    def score_value(self, Model_cy model, Value value,
              numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == self.ptr.groups.size(), \
            "scores_accum != len(classifier)"
        cdef VectorFloat scores
        vector_float_from_ndarray(scores, scores_accum)
        self.ptr.score_value(model.ptr[0], value, scores, get_rng()[0])
        vector_float_to_ndarray(scores, scores_accum)


cdef class Model_cy:
    cdef Model_cc * ptr
    def __cinit__(self):
        self.ptr = new Model_cc()
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        self.ptr.gamma = raw['gamma']
        self.ptr.alpha = raw['alpha']
        self.ptr.betas.clear()
        cdef dict raw_betas = raw['betas']
        self.ptr.betas.resize(len(raw_betas))
        cdef str i
        cdef float beta
        cdef double beta0 = 1.0
        for i, beta in raw_betas.iteritems():
            self.ptr.betas[int(i)] = beta
            beta0 -= beta
        self.ptr.beta0 = beta0

    def dump(self):
        cdef dict betas = {}
        cdef int i
        for i in xrange(self.ptr.betas.size()):
            betas[str(i)] = self.ptr.betas[i]
        return {
            'gamma': float(self.ptr.gamma),
            'alpha': float(self.ptr.alpha),
            'betas': betas,
        }

    #-------------------------------------------------------------------------
    # Sampling

    def sample_value(self, Group group):
        cdef Value value = self.ptr.sample_value(group.ptr[0], get_rng()[0])
        return value

    def sample_group(self, int size):
        cdef Group group = Group()
        cdef Model_cc.Sampler sampler
        self.ptr.sampler_init(sampler, group.ptr[0], get_rng()[0])
        cdef list result = []
        cdef int i
        cdef Value value
        for i in xrange(size):
            value = self.ptr.sampler_eval(sampler, get_rng()[0])
            result.append(value)
        return result

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, Value value):
        return self.ptr.score_value(group.ptr[0], value, get_rng()[0])

    def score_group(self, Group group):
        return self.ptr.score_group(group.ptr[0], get_rng()[0])

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {
                'gamma': 0.5,
                'alpha': 0.5,
                'betas': {  # beta0 must be zero for unit tests
                    '0': 0.25,
                    '1': 0.5,
                    '2': 0.25,
                },
            },
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletProcessDiscrete(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group

    Classifier = Classifier


Model = DirichletProcessDiscrete
