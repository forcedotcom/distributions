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
from libcpp.utility cimport pair
cimport numpy
numpy.import_array()
from cython import address
from cython.operator cimport dereference as deref, preincrement as inc
from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng
from distributions.lp.vector cimport VectorFloat, vector_float_to_ndarray
from distributions.mixins import SharedIoMixin


cdef extern from 'distributions/mixture.hpp':
    cppclass IdSet \
            "std::unordered_set<size_t, distributions::TrivialHash<size_t>>":
        cppclass iterator "const_iterator":
            size_t & operator*()
            iterator operator++() nogil
            bint operator!=(iterator) nogil
        iterator begin() nogil
        iterator end() nogil


cdef extern from 'distributions/clustering.hpp':
    cppclass Assignments "distributions::Clustering<int>::Assignments":
        Assignments() nogil except +
        cppclass iterator:
            pair[int, int] & operator*() nogil
            iterator operator++() nogil
            bint operator!=(iterator) nogil
        int & operator[](int) nogil
        iterator begin() nogil
        iterator end() nogil

    cdef vector[int] count_assignments_cc \
            "distributions::Clustering<int>::count_assignments" \
            (Assignments & assignments) nogil except +

    cppclass PitmanYor_cc "distributions::Clustering<int>::PitmanYor":
        float alpha
        float d
        vector[int] sample_assignments(int size, rng_t & rng) nogil except +
        cppclass Mixture:
            size_t size "counts().size" () nogil except +
            IdSet.iterator empty_groupids_begin \
                    "empty_groupids().begin" () nogil except +
            IdSet.iterator empty_groupids_end \
                    "empty_groupids().end" () nogil except +
            void set_counts "counts() = " (vector[int] &) nogil except +
            void init (PitmanYor_cc &) nogil except +
            bint add_value (PitmanYor_cc &, size_t) nogil except +
            bint remove_value (PitmanYor_cc &, size_t) nogil except +
            void score_value (PitmanYor_cc &, VectorFloat &) nogil except +
        float score_counts(vector[int] & counts) nogil except +
        float score_add_value (
                int group_size,
                int nonempty_group_count,
                int sample_size,
                int empty_group_count) nogil except +
        float score_remove_value (
                int group_size,
                int nonempty_group_count,
                int sample_size,
                int empty_group_count) nogil except +

    cppclass LowEntropy_cc "distributions::Clustering<int>::LowEntropy":
        int dataset_size
        vector[int] sample_assignments(int size, rng_t & rng) nogil except +
        cppclass Mixture:
            size_t size "counts().size" () nogil except +
            IdSet.iterator empty_groupids_begin \
                    "empty_groupids().begin" () nogil except +
            IdSet.iterator empty_groupids_end \
                    "empty_groupids().end" () nogil except +
            void set_counts "counts() = " (vector[int] &) nogil except +
            void init (LowEntropy_cc &) nogil except +
            bint add_value (LowEntropy_cc &, size_t) nogil except +
            bint remove_value (LowEntropy_cc &, size_t) nogil except +
            void score_value (LowEntropy_cc &, VectorFloat &) nogil except +
        float score_counts(vector[int] & counts) nogil except +
        float score_add_value (
                int group_size,
                int nonempty_group_count,
                int sample_size,
                int empty_group_count) nogil except +
        float score_remove_value (
                int group_size,
                int nonempty_group_count,
                int sample_size,
                int empty_group_count) nogil except +


cpdef list count_assignments(dict assignments):
    cdef Assignments assignments_cc
    cdef int value_id
    cdef int group_id
    for value_id, group_id in assignments.iteritems():
        assignments_cc[value_id] = group_id
    cdef list counts = count_assignments_cc(assignments_cc)
    return counts


cdef dict dump_assignments(Assignments & assignments):
    cdef dict raw = {}
    cdef Assignments.iterator i = assignments.begin()
    while i != assignments.end():
        assignments[deref(i).first] = deref(i).second
        inc(i)
    return raw


#-----------------------------------------------------------------------------
# Pitman-Yor

cdef class PitmanYor_cy:
    cdef PitmanYor_cc * ptr
    def __cinit__(self):
        self.ptr = new PitmanYor_cc()
    def __dealloc__(self):
        del self.ptr

    def __init__(self, **kwargs):
        if kwargs:
            self.load(kwargs)
        else:
            self.ptr.alpha = 1.0
            self.ptr.d = 0.0

    property alpha:
        def __get__(self):
            return self.ptr.alpha

    property d:
        def __get__(self):
            return self.ptr.d

    def load(self, dict raw):
        cdef float alpha = raw['alpha']
        cdef float d = raw['d']
        assert 0 < alpha
        assert 0 <= d and d < 1
        self.ptr.alpha = alpha
        self.ptr.d = d

    def dump(self):
        return {
            'alpha': self.ptr.alpha,
            'd': self.ptr.d,
        }

    def sample_assignments(self, int size):
        cdef list assignments = self.ptr.sample_assignments(size, get_rng()[0])
        return assignments

    def score_counts(self, list counts):
        cdef vector[int] counts_cc = counts
        cdef float score = self.ptr.score_counts(counts_cc)
        return score

    def score_add_value(
            self,
            int group_size,
            int nonempty_group_count,
            int sample_size,
            int empty_group_count=1):
        return self.ptr.score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    def score_remove_value(
            self,
            int group_size,
            int nonempty_group_count,
            int sample_size,
            int empty_group_count=1):
        return self.ptr.score_remove_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    EXAMPLES = [
        {'alpha': 1., 'd': 0.},
        {'alpha': 1., 'd': 0.1},
        {'alpha': 1., 'd': 0.9},
        {'alpha': 10., 'd': 0.1},
        {'alpha': 0.1, 'd': 0.1},
    ]


cdef class PitmanYorMixture:
    cdef PitmanYor_cc.Mixture * ptr
    def __cinit__(self):
        self.ptr = new PitmanYor_cc.Mixture()
    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.size()

    property empty_groupids:
        def __get__(self):
            cdef PitmanYor_cc.Mixture * ptr = self.ptr
            cdef IdSet.iterator i = ptr.empty_groupids_begin()
            cdef IdSet.iterator end = ptr.empty_groupids_end()
            while i != end:
                yield deref(i)
                inc(i)

    def init(self, PitmanYor_cy model, list counts):
        cdef vector[int] counts_cc = counts
        self.ptr.set_counts(counts_cc)
        self.ptr.init(model.ptr[0])

    def add_value(self, PitmanYor_cy model, int groupid):
        return self.ptr.add_value(model.ptr[0], groupid)

    def remove_value(self, PitmanYor_cy model, int groupid):
        return self.ptr.remove_value(model.ptr[0], groupid)

    def score_value(
            self,
            PitmanYor_cy model,
            numpy.ndarray[numpy.float32_t, ndim=1] scores):
        cdef VectorFloat scores_cc
        scores_cc.resize(self.ptr.size())
        self.ptr.score_value(model.ptr[0], scores_cc)
        vector_float_to_ndarray(scores_cc, scores)


class PitmanYor(PitmanYor_cy, SharedIoMixin):

    def protobuf_load(self, message):
        self.load({'alpha': message.alpha, 'd': message.d})

    def protobuf_dump(self, message):
        dumped = self.dump()
        message.Clear()
        message.alpha = dumped['alpha']
        message.d = dumped['d']

    Mixture = PitmanYorMixture


#-----------------------------------------------------------------------------
# Low Entropy

cdef class LowEntropy_cy:
    cdef LowEntropy_cc * ptr
    def __cinit__(self):
        self.ptr = new LowEntropy_cc()
    def __dealloc__(self):
        del self.ptr

    def __init__(self, **kwargs):
        if kwargs:
            self.load(kwargs)
        else:
            self.ptr.dataset_size = 0

    property dataset_size:
        def __get__(self):
            return self.ptr.dataset_size

    def load(self, dict raw):
        cdef int dataset_size = raw['dataset_size']
        assert dataset_size >= 0
        self.ptr.dataset_size = dataset_size

    def dump(self):
        return {'dataset_size': self.ptr.dataset_size}

    def sample_assignments(self, int size):
        cdef list assignments = self.ptr.sample_assignments(size, get_rng()[0])
        return assignments

    def score_counts(self, list counts):
        cdef vector[int] counts_cc = counts
        cdef float score = self.ptr.score_counts(counts_cc)
        return score

    def score_add_value(
            self,
            int group_size,
            int nonempty_group_count,
            int sample_size,
            int empty_group_count=1):
        return self.ptr.score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    def score_remove_value(
            self,
            int group_size,
            int nonempty_group_count,
            int sample_size,
            int empty_group_count=1):
        return self.ptr.score_remove_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    EXAMPLES = [
        {'dataset_size': 5},
        {'dataset_size': 10},
        {'dataset_size': 100},
        {'dataset_size': 1000},
    ]


cdef class LowEntropyMixture:
    cdef LowEntropy_cc.Mixture * ptr
    def __cinit__(self):
        self.ptr = new LowEntropy_cc.Mixture()
    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.size()

    property empty_groupids:
        def __get__(self):
            cdef LowEntropy_cc.Mixture * ptr = self.ptr
            cdef IdSet.iterator i = ptr.empty_groupids_begin()
            cdef IdSet.iterator end = ptr.empty_groupids_end()
            while i != end:
                yield deref(i)
                inc(i)

    def init(self, LowEntropy_cy model, list counts):
        cdef vector[int] counts_cc = counts
        self.ptr.set_counts(counts_cc)
        self.ptr.init(model.ptr[0])

    def add_value(self, LowEntropy_cy model, int groupid):
        return self.ptr.add_value(model.ptr[0], groupid)

    def remove_value(self, LowEntropy_cy model, int groupid):
        return self.ptr.remove_value(model.ptr[0], groupid)

    def score_value(
            self,
            LowEntropy_cy model,
            numpy.ndarray[numpy.float32_t, ndim=1] scores):
        cdef VectorFloat scores_cc
        scores_cc.resize(self.ptr.size())
        self.ptr.score_value(model.ptr[0], scores_cc)
        vector_float_to_ndarray(scores_cc, scores)


class LowEntropy(LowEntropy_cy, SharedIoMixin):

    def protobuf_load(self, message):
        self.load({'dataset_size': message.dataset_size})

    def protobuf_dump(self, message):
        dumped = self.dump()
        message.Clear()
        message.datset_size = dumped['dataset_size']

    Mixture = LowEntropyMixture
