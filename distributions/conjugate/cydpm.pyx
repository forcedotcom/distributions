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
cimport numpy as np
np.import_array()

from libc.math cimport log
#from libcpp.map cimport map as fastmap
#from unordered_map cimport unordered_map
from scimath cimport gammaln, dirichlet_draw, categorical_draw
from distributions.util import stick
from sparse_counter cimport SparseCounter, SparseCounter_iterator
from cython.operator cimport dereference as deref, preincrement as inc


cpdef int OTHER = -1


cdef class HP:
    cdef double gamma
    cdef double alpha
    cdef double beta0
    #cdef np.ndarray[double, ndim=1] betas
    cdef np.ndarray betas
    def __init__(self, gamma, alpha, beta0, betas):
        self.gamma = gamma
        self.alpha = alpha
        self.beta0 = beta0
        cdef int size = len(betas)
        self.betas = np.array(betas, dtype=np.double)
        cdef int i
        for i in xrange(size):
            self.betas[i] = betas[i]

cdef class SS:
    cdef SparseCounter * counter
    def __cinit__(self):
        self.counter = new SparseCounter()
    def __dealloc__(self):
        del self.counter


def create_ss(ss=None, p=None):
    cdef SS result = SS()
    cdef SparseCounter * counter = result.counter
    if ss is not None:
        for i, count in ss['counts'].iteritems():
            counter.init_count(int(i), count)
    return result


def dump_ss(SS ss):
    counts = {}
    cdef SparseCounter * counter = ss.counter
    cdef SparseCounter_iterator it = counter.begin()
    cdef SparseCounter_iterator end = counter.end()
    while it != end:
        counts[str(deref(it).first)] = deref(it).second
        inc(it)
    return {'counts': counts}


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1.0, 1.0, 1.0, [])
    else:
        betas = [hp['betas'][str(i)] for i in xrange(len(hp['betas']))]
        return HP(hp['gamma'], hp['alpha'], hp['beta0'], betas)


def dump_hp(HP hp):
    return {
        'gamma': hp.gamma,
        'alpha': hp.alpha,
        'beta0': hp.beta0,
        'betas': {str(i): beta for i, beta in enumerate(hp.betas)},
        }


def realize_hp(HP hp, float tolerance=1e-3):
    """
    Converts betas to a full (approximate) sample from a DP
    """
    if hp.beta0 > 0:
        hp.beta0 = 0.
        betas = stick(hp.gamma, tolerance).values()
        hp.betas = np.array(betas, dtype=np.double)


def add_data(SS ss, int y):
    assert y != OTHER, 'tried to add OTHER to suffstats'
    ss.counter.add(y)


def remove_data(SS ss, int y):
    assert y != OTHER, 'tried to remove OTHER to suffstats'
    ss.counter.remove(y)


cdef _sample_post(HP hp, SS ss):
    cdef SparseCounter * counter = ss.counter
    cdef int size = hp.betas.shape[0]
    cdef np.ndarray[double, ndim=1] values = np.zeros(size + 1, dtype=np.double)
    values[:-1] = hp.betas * hp.alpha
    values[-1] = hp.beta0 * hp.alpha
    cdef SparseCounter_iterator it = counter.begin()
    cdef SparseCounter_iterator end = counter.end()
    cdef int i
    cdef int count
    while it != end:
        i = deref(it).first
        count = deref(it).second
        values[i] += count
        inc(it)
    cdef np.ndarray[double, ndim=1] post = np.zeros(size + 1, dtype=np.double)
    dirichlet_draw(size + 1, <double *> values.data, <double *> post.data)
    return post


def sample_data(HP hp, SS ss):
    cdef int size = hp.betas.shape[0]
    cdef np.ndarray[double, ndim=1] post = _sample_post(hp, ss)
    cdef int index = categorical_draw(size + 1, <double *> post.data)
    if index == size:
        return OTHER
    else:
        return index


def sample_post(HP hp, SS ss):
    return _sample_post(hp, ss)


def generate_post(HP hp, SS ss):
    post = sample_post(hp, ss)
    return {'p': post.tolist()}


cpdef double pred_prob(HP hp, SS ss, int y) except ? 0:
    cdef SparseCounter * counter = ss.counter
    cdef double denom = hp.alpha + counter.get_total()
    cdef double numer
    if y == OTHER:
        numer = hp.beta0 * hp.alpha
    else:
        numer = hp.betas[y] * hp.alpha + counter.get_count(y)
    return log(numer / denom)


def data_prob(HP hp, SS ss):
    assert len(hp.betas), 'betas is empty'
    cdef double score = 0.
    cdef SparseCounter * counter = ss.counter
    cdef SparseCounter_iterator it = counter.begin()
    cdef SparseCounter_iterator end = counter.end()
    cdef int i
    cdef int count
    cdef double prior_y
    while it != end:
        i = deref(it).first
        count = deref(it).second
        prior_y = hp.betas[i] * hp.alpha
        score += gammaln(prior_y + count) - gammaln(prior_y)
        inc(it)
    score += gammaln(hp.alpha) - gammaln(hp.alpha + counter.get_total())

    return score


cpdef add_pred_probs(
        HP hp,
        list ss,
        int y,
        np.ndarray[double, ndim=1] scores):
    """
    Vectorize over it: scores[it] += pred_prob(hp[it], ss[it], y)
    """
    cdef int size = len(scores)
    assert len(ss) == size
    cdef int i
    for i in xrange(size):
        scores[i] += pred_prob(hp, ss[i], y)
    return None  # workaround 'except *' incompatibility with 'cpdef void'
