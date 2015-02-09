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

import numpy as np
import math

from scipy.special import multigammaln
from distributions.dbg.random import (
    score_student_t,
    sample_normal_inverse_wishart,
)
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin

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


class Shared(SharedMixin, SharedIoMixin):

    def dim(self):
        return self.mu.shape[0]

    def plus_group(self, group):
        """
        \cite{murphy2007conjugate}, Eq. 251-254
        """
        mu0, kappa0, psi0, nu0 = self.mu, self.kappa, self.psi, self.nu
        n, sum_x, sum_xxT = group.count, group.sum_x, group.sum_xxT
        xbar = sum_x / n if n else np.zeros(self.dim())
        mu_n = kappa0 / (kappa0 + n) * mu0 + n / (kappa0 + n) * xbar
        kappa_n = kappa0 + n
        nu_n = nu0 + n
        diff = xbar - mu0
        C_n = (
            sum_xxT
            - np.outer(sum_x, xbar)
            - np.outer(xbar, sum_x)
            + n * np.outer(xbar, xbar)
        )
        psi_n = psi0 + C_n + kappa0 * n / (kappa0 + n) * np.outer(diff, diff)
        post = Shared()
        post.mu, post.kappa, post.psi, post.nu = mu_n, kappa_n, psi_n, nu_n
        return post

    def load(self, raw):
        self.mu = raw['mu'].copy()
        assert len(self.mu.shape) == 1
        self.kappa = float(raw['kappa'])
        assert self.kappa > 0.
        self.psi = raw['psi'].copy()
        assert self.mu.shape[0] == self.psi.shape[0]
        assert self.psi.shape[0] == self.psi.shape[1]
        self.nu = float(raw['nu'])
        assert self.nu >= self.dim()

    def dump(self):
        return {
            'mu': self.mu.copy(),
            'kappa': self.kappa,
            'psi': self.psi.copy(),
            'nu': self.nu,
        }

    def protobuf_load(self, message):
        self.mu = np.array(message.mu, dtype=np.float)
        self.kappa = message.kappa
        self.psi = np.array(message.psi, dtype=np.float)
        D = self.dim()
        assert self.psi.shape[0] == (D * D)
        self.psi = self.psi.reshape((D, D))
        self.nu = message.nu
        assert self.nu >= self.dim()

    def protobuf_dump(self, message):
        message.Clear()
        for x in self.mu:
            message.mu.append(x)
        message.kappa = self.kappa
        for x in self.psi:
            for y in x:
                message.psi.append(y)
        message.nu = self.nu


class Group(GroupIoMixin):

    def init(self, shared):
        self.count = 0
        self.sum_x = np.zeros(shared.dim())
        self.sum_xxT = np.zeros((shared.dim(), shared.dim()))

    def add_value(self, shared, value):
        self.count += 1
        self.sum_x += value
        self.sum_xxT += np.outer(value, value)

    def add_repeated_value(self, shared, value, count):
        self.count += count
        self.sum_x += (count * value)
        self.sum_xxT += (count * np.outer(value, value))

    def remove_value(self, shared, value):
        self.count -= 1
        self.sum_x -= value
        self.sum_xxT -= np.outer(value, value)

    def merge(self, shared, source):
        self.count += source.count
        self.sum_x += source.sum_x
        self.sum_xxT += source.sum_xxT

    def score_value(self, shared, value):
        """
        \cite{murphy2007conjugate}, Eq. 258
        """
        post = shared.plus_group(self)
        mu_n, kappa_n, psi_n, nu_n = post.mu, post.kappa, post.psi, post.nu
        dof = nu_n - shared.dim() + 1.
        sigma_n = psi_n * (kappa_n + 1.) / (kappa_n * dof)
        return score_student_t(value, dof, mu_n, sigma_n)

    def score_data(self, shared):
        """
        \cite{murphy2007conjugate}, Eq. 266
        """
        kappa0, psi0, nu0 = shared.kappa, shared.psi, shared.nu
        post = shared.plus_group(self)
        kappa_n, psi_n, nu_n = post.kappa, post.psi, post.nu
        n = self.count
        D = shared.dim()
        return (
            multigammaln(nu_n / 2., D)
            + nu0 / 2. * np.log(np.linalg.det(psi0))
            - (n * D / 2.) * np.log(math.pi)
            - multigammaln(nu0 / 2., D)
            - nu_n / 2. * np.log(np.linalg.det(psi_n))
            + D / 2. * np.log(kappa0 / kappa_n))

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        self.count = int(raw['count'])
        assert self.count >= 0
        self.sum_x = raw['sum_x'].copy()
        self.sum_xxT = raw['sum_xxT'].copy()
        D = self.sum_x.shape[0]
        assert self.sum_xxT.shape == (D, D)

    def dump(self):
        return {
            'count': self.count,
            'sum_x': self.sum_x.copy(),
            'sum_xxT': self.sum_xxT.copy(),
        }

    def protobuf_load(self, message):
        self.count = message.count
        self.sum_x = np.array(message.sum_x, dtype=np.float)
        self.sum_xxT = np.array(message.sum_xxT, dtype=np.float)
        D = self.sum_x.shape[0]
        self.sum_xxT = self.sum_xxT.reshape((D, D))

    def protobuf_dump(self, message):
        message.Clear()
        message.count = self.count
        for x in self.sum_x:
            message.sum_x.append(x)
        for x in self.sum_xxT:
            for y in x:
                message.sum_xxT.append(y)


class Sampler(object):
    def init(self, shared, group=None):
        if group is not None:
            shared = shared.plus_group(group)
        self.mu, self.sigma = sample_normal_inverse_wishart(
            shared.mu, shared.kappa, shared.psi, shared.nu)

    def eval(self, shared):
        return np.random.multivariate_normal(self.mu, self.sigma)


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
