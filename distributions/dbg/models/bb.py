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

from distributions.dbg.special import log, gammaln
from distributions.dbg.random import sample_bernoulli, sample_beta
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'BetaBernoulli'
EXAMPLES = [
    {
        'shared': {'alpha': 0.5, 'beta': 2.0},
        'values': [False, False, True, False, True, True, False, False],
    },
    {
        'shared': {'alpha': 10.5, 'beta': 0.5},
        'values': [False, False, False, False, False, False, False, True],
    },
]
Value = bool


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.alpha = None
        self.beta = None

    def load(self, raw):
        self.alpha = float(raw['alpha'])
        self.beta = float(raw['beta'])

    def dump(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
        }

    def protobuf_load(self, message):
        self.alpha = float(message.alpha)
        self.beta = float(message.beta)

    def protobuf_dump(self, message):
        message.alpha = self.alpha
        message.beta = self.beta


class Group(GroupIoMixin):
    def __init__(self):
        self.heads = None
        self.tails = None

    def init(self, shared):
        self.heads = 0
        self.tails = 0

    def add_value(self, shared, value):
        if value:
            self.heads += 1
        else:
            self.tails += 1

    def add_repeated_value(self, shared, value, count):
        if value:
            self.heads += count
        else:
            self.tails += count

    def remove_value(self, shared, value):
        if value:
            self.heads -= 1
        else:
            self.tails -= 1

    def merge(self, shared, source):
        self.heads += source.heads
        self.tails += source.tails

    def score_value(self, shared, value):
        """
        \cite{wallach2009rethinking} Eqn 4.
        McCallum, et. al, 'Rething LDA: Why Priors Matter'
        """
        heads = shared.alpha + self.heads
        tails = shared.beta + self.tails
        numer = heads if value else tails
        denom = heads + tails
        return log(numer / denom)

    def score_data(self, shared):
        """
        \cite{jordan2001more} Eqn 22.
        Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """
        alpha = shared.alpha + self.heads
        beta = shared.beta + self.tails
        score = gammaln(shared.alpha + shared.beta) - gammaln(alpha + beta)
        score += gammaln(alpha) - gammaln(shared.alpha)
        score += gammaln(beta) - gammaln(shared.beta)
        return score

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        self.heads = raw['heads']
        self.tails = raw['tails']

    def dump(self):
        return {
            'heads': self.heads,
            'tails': self.tails,
        }

    def protobuf_load(self, message):
        self.heads = message.heads
        self.tails = message.tails

    def protobuf_dump(self, message):
        message.heads = self.heads
        message.tails = self.tails


class Sampler(object):
    def init(self, shared, group=None):
        if group is None:
            self.p = sample_beta(shared.alpha, shared.beta)
        else:
            alpha = shared.alpha + group.heads
            beta = shared.beta + group.tails
            self.p = sample_beta(alpha, beta)

    def eval(self, shared):
        return sample_bernoulli(self.p)


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
