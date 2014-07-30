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

cimport _bnb
import _bnb

from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'BetaNegativeBinomial'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'beta': 1., 'r': 1},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    },
]
Value = int


cdef class _Shared(_bnb.Shared):
    def load(self, raw):
        self.ptr.alpha = raw['alpha']
        self.ptr.beta = raw['beta']
        self.ptr.r = raw['r']

    def dump(self):
        return {
            'alpha': self.ptr.alpha,
            'beta': self.ptr.beta,
            'r': self.ptr.r,
        }

    def protobuf_load(self, message):
        self.ptr.alpha = message.alpha
        self.ptr.beta = message.beta
        self.ptr.r = message.r

    def protobuf_dump(self, message):
        message.alpha = self.ptr.alpha
        message.beta = self.ptr.beta
        message.r = self.ptr.r


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group(_bnb.Group):
    def load(self, raw):
        self.ptr.count = raw['count']
        self.ptr.sum = raw['sum']

    def dump(self):
        return {
            'count': self.ptr.count,
            'sum': self.ptr.sum,
        }


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_bnb.Sampler):
    pass


Mixture = _bnb.Mixture
sample_group = _bnb.sample_group
