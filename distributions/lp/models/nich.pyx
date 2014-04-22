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
from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng
from distributions.lp.vector cimport (
    VectorFloat,
    vector_float_from_ndarray,
    vector_float_to_ndarray,
)
from distributions.mixins import ComponentModel, Serializable

cimport nich_cc as cc
ctypedef cc.Value Value


cdef class Group:
    cdef cc.Group * ptr
    def __cinit__(self):
        self.ptr = new cc.Group()
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        self.ptr.count = raw['count']
        self.ptr.mean = raw['mean']
        self.ptr.count_times_variance = raw['count_times_variance']

    def dump(self):
        return {
            'count': self.ptr.count,
            'mean': self.ptr.mean,
            'count_times_variance': self.ptr.count_times_variance,
        }

    def init(self, Model_cy model):
        self.ptr.init(model.ptr[0], get_rng()[0])

    def add_value(self, Model_cy model, Value value):
        self.ptr.add_value(model.ptr[0], value, get_rng()[0])

    def remove_value(self, Model_cy model, Value value):
        self.ptr.remove_value(model.ptr[0], value, get_rng()[0])

    def merge(self, Model_cy model, Group source):
        self.ptr.merge(model.ptr[0], source.ptr[0], get_rng()[0])


cdef class Mixture:
    cdef cc.Mixture * ptr
    cdef VectorFloat scores
    def __cinit__(self):
        self.ptr = new cc.Mixture()
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
            "scores_accum != len(mixture)"
        vector_float_from_ndarray(self.scores, scores_accum)
        self.ptr.score_value(model.ptr[0], value, self.scores, get_rng()[0])
        vector_float_to_ndarray(self.scores, scores_accum)


cdef class Model_cy:
    cdef cc.Model * ptr
    def __cinit__(self):
        self.ptr = new cc.Model()
    def __dealloc__(self):
        del self.ptr

    def load(self, dict raw):
        self.ptr.mu = raw['mu']
        self.ptr.kappa = raw['kappa']
        self.ptr.sigmasq = raw['sigmasq']
        self.ptr.nu = raw['nu']

    def dump(self):
        return {
            'mu': self.ptr.mu,
            'kappa': self.ptr.kappa,
            'sigmasq': self.ptr.sigmasq,
            'nu': self.ptr.nu,
        }

    #-------------------------------------------------------------------------
    # Sampling

    def sample_value(self, Group group):
        cdef Value value = cc.sample_value(self.ptr[0], group.ptr[0], get_rng()[0])
        return value

    def sample_group(self, int size):
        cdef Group group = Group()
        cdef cc.Sampler sampler
        sampler.init(self.ptr[0], group.ptr[0], get_rng()[0])
        cdef list result = []
        cdef int i
        cdef Value value
        for i in xrange(size):
            value = sampler.eval(self.ptr[0], get_rng()[0])
            result.append(value)
        return result

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, Group group, Value value):
        return cc.score_value(self.ptr[0], group.ptr[0], value, get_rng()[0])

    def score_group(self, Group group):
        return cc.score_group(self.ptr[0], group.ptr[0], get_rng()[0])

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
            'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
        },
    ]


class NormalInverseChiSq(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = float

    Group = Group

    Mixture = Mixture


Model = NormalInverseChiSq
