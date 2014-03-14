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

from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
cimport numpy
numpy.import_array()
from distributions.lp.random cimport rng_t, global_rng
from distributions.lp.vector cimport VectorFloat
from distributions.mixins import ComponentModel, Serializable

cpdef int MAX_DIM = 256


ctypedef uint32_t count_t
ctypedef int Value


cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    cdef cppclass Model_cc "distributions::DirichletDiscrete<256>":
        int dim
        float alphas[256]
        #cppclass Value
        cppclass Group:
            count_t count_sum
            count_t counts[]
        cppclass Sampler:
            float ps[256]
        cppclass Scorer:
            float alpha_sum
            float alphas[256]
        cppclass Classifier:
            vector[Group] groups
            float alpha_sum
            vector[VectorFloat] scores
            VectorFloat scores_shift
        void group_init (Group &, rng_t &) nogil
        void group_add_value (Group &, Value &, rng_t &) nogil
        void group_remove_value (Group &, Value &, rng_t &) nogil
        void group_merge (Group &, Group &, rng_t &) nogil
        void sampler_init (Sampler &, Group &, rng_t &) nogil
        Value sampler_eval (Sampler &, rng_t &) nogil
        Value sample_value (Group &, rng_t &) nogil
        float score_value (Group &, Value &, rng_t &) nogil
        float score_group (Group &, rng_t &) nogil
        void classifier_init (Classifier &, rng_t &) nogil
        void classifier_add_group (Classifier &, rng_t &) nogil
        void classifier_remove_group (Classifier &, size_t) nogil
        void classifier_add_value \
            (Classifier &, size_t, Value &, rng_t &) nogil
        void classifier_remove_value \
            (Classifier &, size_t, Value &, rng_t &) nogil
        void classifier_score (Classifier &, Value &, float *, rng_t &) nogil

cdef class Group:
    cdef Model_cc.Group * ptr
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.ptr = new Model_cc.Group()
        self.dim = 0
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        counts = raw['counts']
        self.dim = len(counts)
        self.ptr.count_sum = 0
        cdef int i
        for i in xrange(self.dim):
            self.ptr.count_sum += counts[i]
            self.ptr.counts[i] = counts[i]

    def dump(self):
        counts = []
        cdef int i
        for i in xrange(self.dim):
            counts.append(self.ptr.counts[i])
        return {'counts': counts}


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


cdef class Model_cy:
    cdef Model_cc * ptr
    def __cinit__(self):
        self.ptr = new Model_cc()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        alphas = raw['alphas']
        cdef int dim = len(alphas)
        self.ptr.dim = dim
        cdef int i
        for i in xrange(dim):
            self.ptr.alphas[i] = float(alphas[i])

    def dump(self):
        alphas = []
        cdef int i
        for i in xrange(self.ptr.dim):
            alphas.append(float(self.ptr.alphas[i]))
        return {'alphas': alphas}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.ptr.dim
        self.ptr.group_init(group.ptr[0], global_rng)

    def group_add_value(self, Group group, Value value):
        self.ptr.group_add_value(group.ptr[0], value, global_rng)

    def group_remove_value(self, Group group, Value value):
        self.ptr.group_remove_value(group.ptr[0], value, global_rng)

    def group_merge(self, Group destin, Group source):
        self.ptr.group_merge(destin.ptr[0], source.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Sampling

    def sample_value(self, Group group):
        cdef Value value = self.ptr.sample_value(group.ptr[0], global_rng)
        return value

    def sample_group(self, int size):
        cdef Group group = Group()
        cdef Model_cc.Sampler sampler
        self.ptr.sampler_init(sampler, group.ptr[0], global_rng)
        cdef list result = []
        cdef int i
        cdef Value value
        for i in xrange(size):
            value = self.ptr.sampler_eval(sampler, global_rng)
            result.append(value)
        return result

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, Value value):
        return self.ptr.score_value(group.ptr[0], value, global_rng)

    def score_group(self, Group group):
        return self.ptr.score_group(group.ptr[0], global_rng)

    #-------------------------------------------------------------------------
    # Classification

    def classifier_init(self, Classifier classifier):
        self.ptr.classifier_init(classifier.ptr[0], global_rng)

    def classifier_add_group(self, Classifier classifier):
        self.ptr.classifier_add_group(classifier.ptr[0], global_rng)

    def classifier_remove_group(self, Classifier classifier, int groupid):
        self.ptr.classifier_remove_group(classifier.ptr[0], groupid)

    def classifier_add_value(
            self,
            Classifier classifier,
            int groupid,
            Value value):
        self.ptr.classifier_add_value(
            classifier.ptr[0],
            groupid,
            value,
            global_rng)

    def classifier_remove_value(
            self,
            Classifier classifier,
            int groupid,
            Value value):
        self.ptr.classifier_remove_value(
            classifier.ptr[0],
            groupid,
            value,
            global_rng)

    def classifier_score(
            self,
            Classifier classifier,
            Value value,
            numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == classifier.ptr.groups.size(), \
            "scores_accum != len(classifier)"
        cdef float * data = <float *> scores_accum.data
        self.ptr.classifier_score(
            classifier.ptr[0],
            value,
            data,
            global_rng)

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alphas': [1.0, 4.0]},
            'values': [0, 1, 1, 1, 1, 0, 1],
        },
        {
            'model': {'alphas': [0.5, 0.5, 0.5, 0.5]},
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletDiscrete(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group

    Classifier = Classifier


Model = DirichletDiscrete
