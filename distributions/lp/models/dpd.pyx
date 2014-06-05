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

cimport _dpd
import _dpd

from cython.operator cimport dereference as deref, preincrement as inc
from distributions.sparse_counter cimport SparseCounter
from distributions.mixins import GroupIoMixin, SharedIoMixin


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


cdef class _Shared(_dpd.Shared):
    def load(self, dict raw):
        self.ptr.gamma = raw['gamma']
        self.ptr.alpha = raw['alpha']
        self.ptr.betas.clear()
        cdef dict raw_betas = raw['betas']
        cdef str i
        cdef float beta
        cdef double beta0 = 1.0
        for i, beta in raw_betas.iteritems():
            self.ptr.betas.get(int(i)) = beta
            beta0 -= beta
        self.ptr.beta0 = beta0

    def dump(self):
        cdef dict betas = {}
        cdef int i
        for i in xrange(self.ptr.betas.size()):
            betas[str(i)] = self.ptr.betas.get(i)
        return {
            'gamma': float(self.ptr.gamma),
            'alpha': float(self.ptr.alpha),
            'betas': betas,
        }

    def load_protobuf(self, message):
        self.ptr.gamma = message.gamma
        self.ptr.alpha = message.alpha
        self.ptr.betas.clear()
        cdef int size = len(message.betas)
        cdef int i
        cdef float beta
        cdef double beta0 = 1.0
        for i in xrange(size):
            beta = message.betas[i]
            self.ptr.betas.add(i, beta)
            beta0 -= beta
        self.ptr.beta0 = beta0

    def dump_protobuf(self, message):
        message.Clear()
        message.gamma = self.ptr.gamma
        message.alpha = self.ptr.alpha
        cdef SparseCounter.iterator it = self.ptr.betas.begin()
        cdef SparseCounter.iterator end = self.ptr.betas.end()
        int value
        while it != end:
            value = deref(it).first
            message.keys.append(value)
            message.betas.append(deref(it).second)
            message.counts.append(self.ptr.counts.get_count(value))


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(_dpd.Group):
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


class Sampler(_dpd.Sampler):
    pass


Mixture = _dpd.Mixture
sample_value = _dpd.sample_value
sample_group = _dpd.sample_group
