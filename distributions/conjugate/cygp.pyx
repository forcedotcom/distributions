# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from libc.math cimport sqrt, log
from scimath cimport gammaln, logfactorial, poisson_draw, gamma_draw
import numpy
cimport numpy


cdef class SS:
    cdef int n
    cdef int sum
    cdef double log_prod
    def __init__(self, n, sum, log_prod):
        self.n = n
        self.sum = sum
        self.log_prod = log_prod


cdef class HP:
    cdef double alpha
    cdef double beta
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


def create_ss(ss=None, p=None):
    if ss is None:
        return SS(0, 0, 0.)
    else:
        return SS(ss['n'], ss['sum'], ss['log_prod'])


def dump_ss(SS ss):
    return {
        'n': ss.n,
        'sum': ss.sum,
        'log_prod': ss.log_prod,
        }


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1., 1.)
    else:
        return HP(hp['alpha'], hp['beta'])


def dump_hp(HP hp):
    return {
        'alpha': hp.alpha,
        'beta': hp.beta,
        }


def add_data(SS ss, int y):
    ss.n += 1
    ss.sum += y
    ss.log_prod += logfactorial(y)


def remove_data(SS ss, int y):
    ss.n -= 1
    ss.sum -= y
    ss.log_prod -= logfactorial(y)


cdef HP _intermediates(HP hp, SS ss):
    cdef double alpha_n = hp.alpha + ss.sum
    cdef double beta_n = 1. / (ss.n + 1. / hp.beta)
    return HP(alpha_n, beta_n)


cpdef sample_post(HP hp, SS ss):
    cdef HP z = _intermediates(hp, ss)
    return gamma_draw(z.alpha, z.beta)


def generate_post(HP hp, SS ss):
    return {'lambda': sample_post(hp, ss)}


def sample_data(HP hp, SS ss):
    mu = sample_post(hp, ss)
    return poisson_draw(mu)


cpdef double pred_prob(HP hp, SS ss, int y):
    cdef HP z = _intermediates(hp, ss)
    return gammaln(z.alpha + y) - gammaln(z.alpha) - \
        z.alpha * log(z.beta) + \
        (z.alpha + y) * log(1. / (1. + 1. / z.beta)) - \
        logfactorial(y)


def data_prob(HP hp, SS ss):
    cdef HP z = _intermediates(hp, ss)
    return gammaln(z.alpha) - gammaln(hp.alpha) + \
        z.alpha * log(z.beta) - hp.alpha * log(hp.beta) - \
        ss.log_prod


cpdef add_pred_probs(
        HP hp,
        list ss,
        int y,
        numpy.ndarray[double, ndim=1] scores):
    """
    Vectorize over i: scores[i] += pred_prob(hp[i], ss[i], y)
    """
    cdef int size = len(scores)
    assert len(ss) == size
    cdef int i
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
