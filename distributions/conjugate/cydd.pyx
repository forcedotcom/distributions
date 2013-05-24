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

import numpy as np

from libc.math cimport log
from scimath cimport gammaln, dirichlet_draw, categorical_draw
import numpy
cimport numpy

DEFAULT_DIMENSIONS = 2


cdef class HP:
    cdef double[256] alphas
    cdef int D
    def __init__(self, alphas):
        self.D = len(alphas)
        for i in xrange(self.D):
            self.alphas[i] = alphas[i]


cdef class SS:
    cdef int[256] counts
    cdef int D
    def __init__(self, counts):
        self.D = len(counts)
        for i in xrange(self.D):
            self.counts[i] = counts[i]


def create_ss(ss=None, p=None):
    if ss is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return SS(np.zeros(D, dtype=np.int32))
    else:
        return SS(np.array(ss['counts'], dtype=np.int32))


def dump_ss(SS ss):
    return {'counts': [ss.counts[i] for i in xrange(ss.D)]}


def create_hp(hp=None, p=None):
    if hp is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return HP(np.ones(D, dtype=np.float32))
    else:
        return HP(np.array(hp['alphas'], dtype=np.float32))


def dump_hp(HP hp):
    return {'alphas': [hp.alphas[i] for i in xrange(hp.D)]}


def add_data(SS ss, unsigned y):
    ss.counts[y] += 1


def remove_data(SS ss, unsigned y):
    ss.counts[y] -= 1


def sample_data(HP hp, SS ss):
    cdef double[256] ps
    _sample_post(hp, ss, ps)
    return categorical_draw(hp.D, ps)


cdef _sample_post(HP hp, SS ss, double *thetas):
    cdef double[256] alpha_n
    cdef int i
    for i in xrange(hp.D):
        alpha_n[i] = ss.counts[i] + hp.alphas[i]
    dirichlet_draw(hp.D, alpha_n, thetas)


cpdef sample_post(HP hp, SS ss):
    cdef double[256] thetas
    _sample_post(hp, ss, thetas)
    r = np.zeros(hp.D)
    cdef int i
    for i in xrange(hp.D):
        r[i] = thetas[i]
    return r


def generate_post(HP hp, SS ss):
    post = sample_post(hp, ss)
    return {'p': post.tolist()}


cpdef double pred_prob(HP hp, SS ss, unsigned y):
    """
    McCallum, et. al, 'Rething LDA: Why Priors Matter' eqn 4
    """
    cdef double sum = 0.
    cdef int i
    for i in xrange(hp.D):
        sum += ss.counts[i] + hp.alphas[i]
    return log((ss.counts[y] + hp.alphas[y]) / sum)


def data_prob(HP hp, SS ss):
    """
    From equation 22 of Michael Jordan's CS281B/Stat241B
    Advanced Topics in Learning and Decision Making course,
    'More on Marginal Likelihood'
    """
    cdef int i
    cdef double alpha_sum = 0.
    cdef int count_sum = 0
    cdef double sum = 0.
    for i in xrange(hp.D):
        alpha_sum += hp.alphas[i]
    for i in xrange(hp.D):
        count_sum += ss.counts[i]
    for i in xrange(hp.D):
        sum += gammaln(hp.alphas[i] + ss.counts[i]) - gammaln(hp.alphas[i])
    return sum + gammaln(alpha_sum) - gammaln(alpha_sum + count_sum)


cpdef add_pred_probs(
        HP hp,
        list ss,
        unsigned y,
        numpy.ndarray[double, ndim=1] scores):
    """
    Vectorize over i: scores[i] += pred_prob(hp[i], ss[i], y)
    """
    cdef int size = len(scores)
    assert len(ss) == size
    cdef int i
    for i in xrange(size):
        scores[i] += pred_prob(hp, ss[i], y)
