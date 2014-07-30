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
from distributions.sparse_counter cimport SparseCounter, SparseFloat
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
            'gamma': 0.5,
            'alpha': 0.5,
            'betas': {
                0: 0.25,
                7: 0.5,
                8: 0.25,
            },
            'counts': {
                0: 1,
                7: 2,
                8: 4,
            },
        },
        'values': [0, 7, 0, 8, 0, 7, 0],
    },
    {
        'shared': {
            'gamma': 2.0,
            'alpha': 2.0,
            'betas': {},
            'counts': {},
        },
        'values': [5, 4, 3, 2, 1, 0, 3, 2, 1],
    },
]
Value = int


cdef class _Shared(_dpd.Shared):
    def load(self, dict raw):
        self.ptr.gamma = raw['gamma']
        self.ptr.alpha = raw['alpha']
        self.ptr.betas.clear()
        self.ptr.counts.clear()
        cdef dict raw_betas = raw['betas']
        cdef dict raw_counts = raw['counts']
        cdef int value
        cdef float beta
        cdef double beta0 = 1.0
        for value, beta in raw_betas.iteritems():
            self.ptr.betas.add(int(value), beta)
            beta0 -= beta
        self.ptr.beta0 = beta0
        cdef int count
        for value, count in raw_counts.iteritems():
            self.ptr.counts.add(int(value), count)

    def dump(self):
        cdef dict betas = {}
        cdef dict counts = {}
        cdef SparseFloat.iterator it = self.ptr.betas.begin()
        cdef SparseFloat.iterator end = self.ptr.betas.end()
        cdef int value
        while it != end:
            value = deref(it).first
            betas[value] = float(deref(it).second)
            counts[value] = int(self.ptr.counts.get_count(value))
            inc(it)
        return {
            'gamma': float(self.ptr.gamma),
            'alpha': float(self.ptr.alpha),
            'betas': betas,
            'counts': counts,
        }

    def protobuf_load(self, message):
        self.ptr.gamma = message.gamma
        self.ptr.alpha = message.alpha
        self.ptr.betas.clear()
        self.ptr.counts.clear()
        cdef int i
        cdef int value
        cdef float beta
        cdef double beta0 = 1.0
        for i in xrange(len(message.betas)):
            value = message.values[i]
            beta = message.betas[i]
            self.ptr.betas.add(value, beta)
            self.ptr.counts.add(value, message.counts[i])
            beta0 -= beta
        self.ptr.beta0 = beta0

    def protobuf_dump(self, message):
        message.Clear()
        message.gamma = self.ptr.gamma
        message.alpha = self.ptr.alpha
        cdef SparseFloat.iterator it = self.ptr.betas.begin()
        cdef SparseFloat.iterator end = self.ptr.betas.end()
        cdef int value
        while it != end:
            value = deref(it).first
            message.values.append(value)
            message.betas.append(deref(it).second)
            message.counts.append(self.ptr.counts.get_count(value))
            inc(it)


#class Shared(_Shared, SharedMixin, SharedIoMixin):
class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(_dpd.Group):
    def load(self, dict raw):
        cdef SparseCounter * counts = & self.ptr.counts
        counts.clear()
        cdef dict raw_counts = raw['counts']
        cdef int value
        cdef int count
        for value, count in raw_counts.iteritems():
            counts.init_count(value, count)

    def dump(self):
        cdef dict counts = {}
        cdef SparseCounter.iterator it = self.ptr.counts.begin()
        cdef SparseCounter.iterator end = self.ptr.counts.end()
        while it != end:
            counts[int(deref(it).first)] = deref(it).second
            inc(it)
        return {'counts': counts}


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_dpd.Sampler):
    pass


Mixture = _dpd.Mixture
sample_group = _dpd.sample_group
