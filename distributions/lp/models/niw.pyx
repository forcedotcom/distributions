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

from distributions._eigen_h cimport VectorXf, MatrixXf
from distributions._eigen cimport (
    to_eigen_vecf,
    to_eigen_matf,
    to_np_1darray,
    to_np_2darray,
)

cimport _niw
import _niw

from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin

import numpy as np

NAME = 'NormalInverseWishart'
EXAMPLES = [
    {
        'shared': {
            'mu': np.zeros(1),
            'kappa': 2.,
            'psi': np.eye(1),
            'nu': 3.,
        },
        'values': [np.array(v) for v in (
            [1.],
            [-2.],
            [-0.2],
            [-0.1],
            [0.8],
            [0.8],
            [-9.],
        )],
    },
    {
        'shared': {
            'mu': np.zeros(2),
            'kappa': 2.,
            'psi': np.eye(2),
            'nu': 3.,
        },
        'values': [np.array(v) for v in (
            [1., 2.],
            [-2., 3.],
            [-0.2, -0.2],
            [-0.1, 0.5],
            [0.8, 0.5],
            [0.8, 0.3],
            [-9., 0.2],
        )],
    },
    {
        'shared': {
            'mu': np.ones(3),
            'kappa': 7.5,
            'psi': np.eye(3),
            'nu': 5.,
        },
        'values': [np.array(v) for v in (
            [1.35, 0.97, 0.88],
            [0.87, 1.74, 2.13],
            [-0.31, 1.48, 1.96],
            [1.18, 0.34, 1.00],
            [1.47, 0.62, -0.10],
            [-0.23, 2.23, 0.99],
            [1.23, 0.98, 0.36],
            [1.97, 0.81, 0.79],
            [0.59, 4.27, 0.44],
        )],
    },
    {
        'shared': {
            'mu': -np.ones(4),
            'kappa': 7.5,
            'psi': np.eye(4),
            'nu': 10.,
        },
        'values': [np.array(v) for v in (
            [0.32, -1.92, -2.13, -0.78],
            [-2.35, -1.98, -0.27, -1.48],
            [-0.54, -1.76, -1.14, 0.24],
            [-0.68, -1.62, -0.76, -1.82],
            [-3.03, 0.54, -1.85, -0.53],
            [0.56, -0.96, -1.00, -2.05],
            [-1.18, -1.52, -1.19, -1.06],
            [0.47, -0.23, -0.99, 0.69],
            [-1.41, -3.18, -3.09, -1.93],
        )],
    },
]
Value = np.ndarray


cdef class _Shared(_niw.Shared):
    def load(self, dict raw):
        # XXX: validate raw['mu'] and raw['psi']
        self.ptr.mu = to_eigen_vecf(raw['mu'])
        self.ptr.kappa = raw['kappa']
        assert raw['psi'] is not None
        self.ptr.psi = to_eigen_matf(raw['psi'])
        self.ptr.nu = raw['nu']

    def dump(self):
        return {
            'mu': to_np_1darray(self.ptr.mu),
            'kappa': self.ptr.kappa,
            'psi': to_np_2darray(self.ptr.psi),
            'nu': self.ptr.nu,
        }

    def protobuf_load(self, message):
        # XXX: build the datastructures directly
        self.ptr.mu = to_eigen_vecf(np.array(message.mu, dtype=float))
        self.ptr.kappa = message.kappa
        D = len(message.mu)
        psi = np.array(message.psi, dtype=float).reshape((D, D))
        self.ptr.psi = to_eigen_matf(psi)
        self.ptr.nu = message.nu

    def protobuf_dump(self, message):
        message.Clear()
        for mu in to_np_1darray(self.ptr.mu):
            message.mu.append(mu)
        message.kappa = self.ptr.kappa
        for x in to_np_2darray(self.ptr.psi):
            for y in x:
                message.psi.append(y)
        message.nu = self.ptr.nu


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group(_niw.Group):
    def load(self, dict raw):
        # XXX: validate raw['sum_x'] and raw['sum_xxT']
        self.ptr.count = raw['count']
        self.ptr.sum_x = to_eigen_vecf(raw['sum_x'])
        assert raw['sum_xxT'] is not None
        self.ptr.sum_xxT = to_eigen_matf(raw['sum_xxT'])

    def dump(self):
        return {
            'count': self.ptr.count,
            'sum_x': to_np_1darray(self.ptr.sum_x),
            'sum_xxT': to_np_2darray(self.ptr.sum_xxT),
        }


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_niw.Sampler):
    pass

sample_group = _niw.sample_group
