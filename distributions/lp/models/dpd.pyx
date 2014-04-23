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
from distributions.mixins import GroupIoMixin, SharedIoMixin

cimport dpd_cc as cc


cdef class __Shared:
    cdef cc.Model * ptr

    def __cinit__(self):
        self.ptr = new cc.Model()

    def __dealloc__(self):
        del self.ptr


cdef class __Group:
    cdef cc.Group * ptr

    def __cinit__(self):
        self.ptr = new cc.Group()

    def __dealloc__(self):
        del self.ptr

    def init(self, __Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_value(self, __Shared shared, cc.Value value):
        self.ptr.add_value(shared.ptr[0], value, get_rng()[0])

    def remove_value(self, __Shared shared, cc.Value value):
        self.ptr.remove_value(shared.ptr[0], value, get_rng()[0])

    def merge(self, __Shared shared, __Group source):
        self.ptr.merge(shared.ptr[0], source.ptr[0], get_rng()[0])


cdef class __Mixture:
    cdef cc.Mixture * ptr
    cdef VectorFloat scores

    def __cinit__(self):
        self.ptr = new cc.Mixture()

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.groups.size()

    def __getitem__(self, int groupid):
        assert groupid < len(self), "groupid out of bounds"
        group = __Group()
        group.ptr[0] = self.ptr.groups[groupid]
        return group

    def append(self, __Group group):
        self.ptr.groups.push_back(group.ptr[0])

    def clear(self):
        self.ptr.groups.clear()

    def init(self, __Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_group(self, __Shared shared):
        self.ptr.add_group(shared.ptr[0], get_rng()[0])

    def remove_group(self, __Shared shared, int groupid):
        self.ptr.remove_group(shared.ptr[0], groupid)

    def add_value(self, __Shared shared, int groupid, cc.Value value):
        self.ptr.add_value(shared.ptr[0], groupid, value, get_rng()[0])

    def remove_value(self, __Shared shared, int groupid, cc.Value value):
        self.ptr.remove_value(shared.ptr[0], groupid, value, get_rng()[0])

    def score_value(self, __Shared shared, cc.Value value,
              numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == self.ptr.groups.size(), \
            "scores_accum != len(mixture)"
        vector_float_from_ndarray(self.scores, scores_accum)
        self.ptr.score_value(shared.ptr[0], value, self.scores, get_rng()[0])
        vector_float_to_ndarray(self.scores, scores_accum)


def __sample_value(__Shared shared, __Group group):
    cdef cc.Value value = cc.sample_value(
        shared.ptr[0], group.ptr[0], get_rng()[0])
    return value

def __sample_group(__Shared shared, int size):
    cdef __Group group = __Group()
    cdef cc.Sampler sampler
    sampler.init(shared.ptr[0], group.ptr[0], get_rng()[0])
    cdef list result = []
    cdef int i
    cdef cc.Value value
    for i in xrange(size):
        value = sampler.eval(shared.ptr[0], get_rng()[0])
        result.append(value)
    return result

def __score_value(__Shared shared, __Group group, cc.Value value):
    return cc.score_value(shared.ptr[0], group.ptr[0], value, get_rng()[0])

def __score_group(__Shared shared, __Group group):
    return cc.score_group(shared.ptr[0], group.ptr[0], get_rng()[0])


##################################################


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
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
Value = int


cdef class _Shared(__Shared):
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


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(__Group):
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


class Group(_Group, GroupIoMixin):
    pass


Mixture = __Mixture
sample_value = __sample_value
sample_group = __sample_group
score_value = __score_value
score_group = __score_group
