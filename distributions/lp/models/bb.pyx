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

cimport _bb
import _bb

from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'BetaBernoulli'
EXAMPLES = [
    {
        'shared': {'alpha': 0.5, 'beta': 2.0},
        'values': [False, False, True, False, True, True, False, False],
    },
    {
        'shared': {'alpha': 10.5, 'beta': 0.5},
        'values': [False, False, False, False, False, False, False, True],
    },
]
Value = bool


cdef class _Shared(_bb.Shared):
    def load(self, raw):
        self.ptr.alpha = float(raw['alpha'])
        self.ptr.beta = float(raw['beta'])

    def dump(self):
        return {
            'alpha': self.ptr.alpha,
            'beta': self.ptr.beta,
        }

    def protobuf_load(self, message):
        self.ptr.alpha = message.alpha
        self.ptr.beta = message.beta

    def protobuf_dump(self, message):
        message.alpha = float(self.ptr.alpha)
        message.beta = float(self.ptr.beta)


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group(_bb.Group):
    def load(self, dict raw):
        self.ptr.heads = raw['heads']
        self.ptr.tails = raw['tails']

    def dump(self):
        return {
            'heads': self.ptr.heads,
            'tails': self.ptr.tails,
        }

class Group(_Group, GroupIoMixin):
    pass


class Sampler(_bb.Sampler):
    pass


Mixture = _bb.Mixture
sample_group = _bb.sample_group
