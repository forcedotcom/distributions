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
import scipy.stats as stats
import scipy.linalg
import math

from scipy.special import multigammaln
from distributions.dbg.special import gammaln
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin

NAME = 'NormalInverseWishart'
EXAMPLES = [
    {
        'shared': {
                'mu': np.zeros(2),
                'kappa': 2.,
                'psi' : np.eye(2),
                'nu' : 3.,
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
]
Value = np.ndarray

def score_mv_student_t(x, nu, mu, sigma):
    """
    Eq. 313
    """
    d = x.shape[0]
    term1 = gammaln(nu/2. + d/2.) - gammaln(nu/2.)
    sigmainv = np.linalg.inv(sigma)
    term2 = -0.5*np.log(np.linalg.det(sigma)) - d/2.*np.log(nu*math.pi)
    diff = x - mu
    term3 = -0.5*(nu+d)*np.log(1. + 1./nu*np.dot(diff, np.dot(sigmainv, diff)))
    return term1 + term2 + term3

def sample_iw(S, nu):
    """
    https://github.com/mattjj/pyhsmm/blob/master/util/stats.py#L140
    """

    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(nu,n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)

def sample_niw(mu0, lambda0, psi0, nu0):
    D, = mu0.shape
    assert psi0.shape == (D,D)
    assert lambda0 > 0.0
    assert nu0 > D - 1
    cov = sample_iw(psi0, nu0)
    mu = np.random.multivariate_normal(mean=mu0, cov=1./lambda0*cov)
    return mu, cov

class Shared(SharedMixin, SharedIoMixin):

    def dim(self):
        return self.mu.shape[0]

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
            'mu' : self.mu.copy(),
            'kappa' : self.kappa,
            'psi' : self.psi.copy(),
            'nu' : self.nu,
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
        self.count +=count
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

    def _post_params(self, shared):
        mu0, lam0, psi0, nu0 = shared.mu, shared.kappa, shared.psi, shared.nu
        n, sum_x, sum_xxT = self.count, self.sum_x, self.sum_xxT
        xbar = sum_x / n if n else np.zeros(shared.dim())
        mu_n = lam0/(lam0 + n)*mu0 + n/(lam0 + n)*xbar
        lam_n = lam0 + n
        nu_n = nu0 + n
        diff = xbar - mu0
        C_n = sum_xxT - np.outer(sum_x, xbar) - np.outer(xbar, sum_x) + n*np.outer(xbar, xbar)
        psi_n = psi0 + C_n + lam0*n/(lam0+n)*np.outer(diff, diff)
        return mu_n, lam_n, psi_n, nu_n

    def score_value(self, shared, value):
        """
        Eq. 258
        """
        mu_n, lam_n, psi_n, nu_n = self._post_params(shared)
        dof = nu_n-shared.dim()+1.
        sigma_n = psi_n*(lam_n+1.)/(lam_n*dof)
        return score_mv_student_t(value, dof, mu_n, sigma_n)

    def score_data(self, shared):
        """
        Eq. 266
        """
        mu0, lam0, psi0, nu0 = shared.mu, shared.kappa, shared.psi, shared.nu
        mu_n, lam_n, psi_n, nu_n = self._post_params(shared)
        n = self.count
        D = shared.dim()
        return multigammaln(nu_n/2., D) \
            + nu0/2.*np.log(np.linalg.det(psi0)) \
            - (n*D/2.)*np.log(math.pi) \
            - multigammaln(nu0/2., D) \
            - nu_n/2.*np.log(np.linalg.det(psi_n)) \
            + D/2.*np.log(lam0/lam_n)

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
        assert self.sum_xxT.shape == (D,D)

    def dump(self):
        return {
            'count' : self.count,
            'sum_x' : self.sum_x.copy(),
            'sum_xxT' : self.sum_xxT.copy(),
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
            mu0, kappa0, psi0, nu0 = group._post_params(shared)
        else:
            mu0, kappa0, psi0, nu0 = shared.mu, shared.kappa, shared.psi, shared.nu
        self.mu, self.sigma = sample_niw(mu0, kappa0, psi0, nu0)

    def eval(self, shared):
        return np.random.multivariate_normal(self.mu, self.sigma)

def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
