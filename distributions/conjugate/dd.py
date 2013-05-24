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

from numpy.random.mtrand import dirichlet
from math import log
from scipy.special import gammaln
import numpy as np

from distributions.util import discrete_draw


DEFAULT_DIMENSIONS = 2


class HP:
    def __init__(self, alphas):
        self.alphas = np.array(alphas, dtype=np.float)


class SS:
    def __init__(self, counts):
        self.counts = np.array(counts, dtype=np.int)


def create_ss(ss=None, p=None):
    if ss is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return SS(np.zeros(D))
    else:
        return SS(ss['counts'])


def dump_ss(ss):
    return {'counts': ss.counts.tolist()}


def create_hp(hp=None, p=None):
    if hp is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return HP(np.ones(D, dtype=np.float32))
    else:
        return HP(np.array(hp['alphas'], dtype=np.float32))


def dump_hp(hp):
    return {'alphas': hp.alphas.tolist()}


def add_data(ss, y):
    ss.counts[y] += 1


def remove_data(ss, y):
    ss.counts[y] -= 1


def sample_data(hp, ss):
    return discrete_draw(sample_post(hp, ss))


def sample_post(hp, ss):
    return dirichlet(ss.counts + hp.alphas)


def generate_post(hp, ss):
    post = sample_post(hp, ss)
    return {'p': post.tolist()}


def pred_prob(hp, ss, y):
    """
    McCallum, et. al, 'Rething LDA: Why Priors Matter' eqn 4
    """
    return log(
        (ss.counts[y] + hp.alphas[y])
        / (sum(ss.counts) + sum(hp.alphas)))


def data_prob(hp, ss):
    """
    From equation 22 of Michael Jordan's CS281B/Stat241B
    Advanced Topics in Learning and Decision Making course,
    'More on Marginal Likelihood'
    """

    a = hp.alphas
    m = ss.counts

    return sum([gammaln(a[k] + m[k]) - gammaln(a[k])
                for k in range(len(ss.counts))]) \
                + gammaln(sum(ak for ak in a)) \
                - gammaln(sum([a[k] + m[k] for k in range(len(ss.counts))]))


def add_pred_probs(hp, ss, y, scores):
    size = len(scores)
    assert len(ss) == size
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
