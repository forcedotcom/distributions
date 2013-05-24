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

from scipy.stats import beta, bernoulli
from scipy.special import betaln
from math import log

"""
A conjugate model in which the prior is the Beta distribution and the
liklihood is the Bernoulli distribution.

Truthy values are treated as a head, and falsy ones as tails.
"""


class SS:
    def __init__(self, heads, tails):
        self.heads = int(heads)
        self.tails = int(tails)


class HP:
    def __init__(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)


def create_ss(ss=None, p=None):
    if ss is None:
        return SS(0, 0)
    else:
        return SS(ss['heads'], ss['tails'])


def dump_ss(ss):
    return vars(ss)


def dump_hp(hp):
    return vars(hp)


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1., 1.)
    else:
        return HP(hp['alpha'], hp['beta'])


def add_data(ss, y):
    if y:
        ss.heads += 1
    else:
        ss.tails += 1


def remove_data(ss, y):
    if y:
        ss.heads -= 1
    else:
        ss.tails -= 1


def sample_data(hp, ss):
    return bernoulli.rvs(sample_post(hp, ss))


def sample_post(hp, ss):
    return beta.rvs(hp.alpha + ss.heads, hp.beta + ss.tails)


def generate_post(hp, ss):
    post = sample_post(hp, ss)
    return {'p': post}


def pred_prob(hp, ss, y):
        # FIXME source?
    d = hp.alpha + ss.heads + hp.beta + ss.tails
    if y:
        return log((ss.heads + hp.alpha) / d)
    else:
        return log((ss.tails + hp.beta) / d)


def data_prob(hp, ss):
    # FIXME source?
    return betaln(hp.alpha + ss.heads, hp.beta + ss.tails) \
        - betaln(hp.alpha, hp.beta)
