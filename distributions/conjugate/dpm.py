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

from distributions.util import discrete_draw, stick


OTHER = -1


class HP:
    def __init__(self, gamma, alpha, beta0, betas):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.betas = np.array(betas, dtype=np.float)  # dense


class SS:
    def __init__(self, counts):
        self.counts = dict(counts)  # sparse
        self.total = sum(counts.itervalues())


def create_ss(ss=None, p=None):
    if ss is None:
        return SS({})
    else:
        counts = {int(i): count
                  for i, count in ss['counts'].iteritems()
                  if count}
        return SS(counts)


def dump_ss(ss):
    counts = {str(i): count for i, count in ss.counts.iteritems() if count}
    return {'counts': counts}


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1.0, 1.0, 1.0, [])
    else:
        betas = [hp['betas'][str(i)] for i in xrange(len(hp['betas']))]
        return HP(hp['gamma'], hp['alpha'], hp['beta0'], betas)


def dump_hp(hp):
    return {
        'gamma': hp.gamma,
        'alpha': hp.alpha,
        'beta0': hp.beta0,
        'betas': {str(i): beta for i, beta in enumerate(hp.betas)},
        }


def realize_hp(hp, tolerance=1e-3):
    """
    Converts betas to a full (approximate) sample from a DP
    """
    if hp.beta0 > 0:
        hp.beta0 = 0.
        betas = stick(hp.gamma, tolerance).values()
        hp.betas = np.array(betas, dtype=np.float)


def add_data(ss, y):
    assert y != OTHER, 'tried to add OTHER to suffstats'
    try:
        ss.counts[y] += 1
    except KeyError:
        ss.counts[y] = 1
    ss.total += 1


def remove_data(ss, y):
    assert y != OTHER, 'tried to remove OTHER to suffstats'
    new_count = ss.counts[y] - 1
    if new_count == 0:
        del ss.counts[y]
    else:
        ss.counts[y] = new_count
    ss.total -= 1


def _sample_post(hp, ss):
    values = (hp.betas * hp.alpha).tolist()
    for i, count in ss.counts.iteritems():
        values[i] += count
    values.append(hp.beta0 * hp.alpha)
    return dirichlet(values)


def sample_data(hp, ss):
    post = _sample_post(hp, ss)
    index = discrete_draw(post)
    if index == len(hp.betas):
        return OTHER
    else:
        return index


def sample_post(hp, ss):
    return _sample_post(hp, ss)


def generate_post(hp, ss):
    post = _sample_post(hp, ss)
    betas, beta0 = post[:-1], post[-1]
    assert beta0 == 0.
    return {'p': betas.tolist()}


def pred_prob(hp, ss, y):
    """
    Adapted from dd.py, which was adapted from:
    McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
    """
    denom = hp.alpha + ss.total
    if y == OTHER:
        numer = hp.beta0 * hp.alpha
    else:
        numer = hp.betas[y] * hp.alpha + ss.counts.get(y, 0)
    return log(numer / denom)


def data_prob(hp, ss):
    assert len(hp.betas), 'betas is empty'
    """
    See dpm.pdf Equation (3)
    """
    score = 0.
    for i, count in ss.counts.iteritems():
        prior_y = hp.betas[i] * hp.alpha
        score += gammaln(prior_y + count) - gammaln(prior_y)
    score += gammaln(hp.alpha) - gammaln(hp.alpha + ss.total)

    return score


def add_pred_probs(hp, ss, y, scores):
    size = len(scores)
    assert len(ss) == size
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
