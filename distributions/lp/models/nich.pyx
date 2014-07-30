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

cimport _nich
import _nich

from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'NormalInverseChiSq'
EXAMPLES = [
    {
        'shared': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    },
]
Value = float


cdef class _Shared(_nich.Shared):
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

    def protobuf_load(self, message):
        self.ptr.mu = message.mu
        self.ptr.kappa = message.kappa
        self.ptr.sigmasq = message.sigmasq
        self.ptr.nu = message.nu

    def protobuf_dump(self, message):
        message.Clear()
        message.mu = self.ptr.mu
        message.kappa = self.ptr.kappa
        message.sigmasq = self.ptr.sigmasq
        message.nu = self.ptr.nu


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group(_nich.Group):
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


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_nich.Sampler):
    pass


Mixture = _nich.Mixture
sample_group = _nich.sample_group
