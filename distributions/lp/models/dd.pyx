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

cimport _dd
import _dd

from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'DirichletDiscrete'
EXAMPLES = [
    {
        'shared': {'alphas': [0.5, 0.5, 0.5, 0.5]},
        'values': [0, 1, 0, 2, 0, 1, 0],
    },
    {
        'shared': {'alphas': [1.0, 4.0]},
        'values': [0, 1, 1, 1, 1, 0, 1],
    },
    {
        'shared': {'alphas': [2.0 / n for n in xrange(1, 21)]},
        'values': range(20),
    },
]
Value = int


cdef class _Shared(_dd.Shared):
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

    def protobuf_load(self, message):
        cdef int dim = len(message.alphas)
        self.ptr.dim = dim
        cdef int i
        for i in xrange(self.ptr.dim):
            self.ptr.alphas[i] = message.alphas[i]

    def protobuf_dump(self, message):
        message.Clear()
        cdef int i
        for i in xrange(self.ptr.dim):
            message.alphas.append(float(self.ptr.alphas[i]))


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group(_dd.Group):
    cdef int dim  # only required for dumping

    def __cinit__(self):
        self.dim = 0

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

    def init(self, _dd.Shared shared):
        self.dim = shared.ptr.dim
        _dd.Group.init(self, shared)


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_dd.Sampler):
    pass


Mixture = _dd.Mixture
sample_group = _dd.sample_group
