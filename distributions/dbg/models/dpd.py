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

"""
\cite{teh2006hierarchical}
"""

from itertools import izip
from distributions.dbg.special import log, gammaln
from distributions.dbg.random import (
    sample_discrete,
    sample_dirichlet,
    sample_beta,
)
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
            'gamma': 0.5,
            'alpha': 0.5,
            'betas': {
                0: 0.25,
                7: 0.5,
                8: 0.25,
            },
            'counts': {
                0: 1,
                7: 2,
                8: 4,
            },
        },
        'values': [0, 7, 0, 8, 0, 7, 0],
    },
    {
        'shared': {
            'gamma': 2.0,
            'alpha': 2.0,
            'betas': {},
            'counts': {},
        },
        'values': [5, 4, 3, 2, 1, 0, 3, 2, 1],
    },
]
OTHER = 0xFFFFFFFF
Value = int


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.gamma = None
        self.alpha = None
        self.beta0 = None
        self.betas = None
        self.counts = None

    def _load_beta0(self):
        self.beta0 = max(0.0, 1.0 - sum(self.betas.itervalues()))
        if not (self.beta0 <= 1):
            raise ValueError('beta0 out of bounds: {}'.format(self.beta0))
        if self.betas:
            min_beta = min(self.betas.itervalues())
            max_beta = max(self.betas.itervalues())
            if not (0 <= min_beta and max_beta <= 1):
                raise ValueError('betas out of bounds: {}'.format(self.betas))

    def load(self, raw):
        self.gamma = float(raw['gamma'])
        self.alpha = float(raw['alpha'])
        self.betas = {
            int(value): float(beta)
            for value, beta in raw['betas'].iteritems()
        }
        self.counts = {
            int(value): int(count)
            for value, count in raw['counts'].iteritems()
        }
        self._load_beta0()

    def dump(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'betas': self.betas.copy(),
            'counts': self.counts.copy(),
        }

    def protobuf_load(self, message):
        assert len(message.betas) == len(message.values), "invalid message"
        assert len(message.counts) == len(message.values), "invalid message"
        self.gamma = float(message.gamma)
        self.alpha = float(message.alpha)
        self.betas = {
            int(value): float(beta)
            for value, beta in izip(message.values, message.betas)
        }
        self.counts = {
            int(value): int(count)
            for value, count in izip(message.values, message.counts)
        }
        self._load_beta0()

    def protobuf_dump(self, message):
        message.Clear()
        message.gamma = self.gamma
        message.alpha = self.alpha
        for value, beta in self.betas.iteritems():
            message.values.append(value)
            message.betas.append(beta)
            message.counts.append(self.counts[value])

    def add_value(self, value):
        assert value != OTHER, 'cannot add OTHER'
        count = self.counts.get(value, 0) + 1
        self.counts[value] = count
        if count == 1:
            beta = self.beta0 * sample_beta(1.0, self.gamma)
            self.beta0 = max(0.0, self.beta0 - beta)
            self.betas[value] = beta

    def remove_value(self, value):
        assert value != OTHER, 'cannot remove OTHER'
        count = self.counts[value] - 1
        if count:
            self.counts[value] = count
        else:
            del self.counts[value]
            self.beta0 += self.betas.pop(value)

    def realize(self):
        max_size = 10000
        min_beta0 = 1e-4
        new_value = 1 + max(self.betas.iterkeys()) if self.betas else 0
        while len(self.betas) < max_size - 1 and self.beta0 > min_beta0:
            self.add_value(new_value)
            new_value += 1
        if self.beta0 > 0:
            self.add_value(new_value)
            self.betas[new_value] += self.beta0
            self.beta0 = 0


class Group(GroupIoMixin):
    def __init__(self):
        self.counts = None
        self.total = None

    def init(self, shared):
        self.counts = {}  # sparse
        self.total = 0

    def add_repeated_value(self, shared, value, count):
        assert value != OTHER, 'cannot add OTHER'
        assert value in shared.betas, 'unknown value: {}'.format(value)
        if count:
            self.total += count
            try:
                count += self.counts[value]
                if count:
                    self.counts[value] = count
                else:
                    del self.counts[value]
            except KeyError:
                self.counts[value] = count

    def add_value(self, shared, value):
        self.add_repeated_value(shared, value, 1)

    def remove_value(self, shared, value):
        self.add_repeated_value(shared, value, -1)

    def score_value(self, shared, value):
        """
        Adapted from dd.py, which was adapted from:
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        denom = shared.alpha + self.total
        if value == OTHER:
            numer = shared.beta0 * shared.alpha
        else:
            count = self.counts.get(value, 0)
            assert count >= 0, "cannot score while in debt"
            numer = shared.betas[value] * shared.alpha + count
        return log(numer / denom)

    def score_data(self, shared):
        assert len(shared.betas), 'betas is empty'
        """
        See doc/dpd.pdf Equation (3)
        """
        score = 0.
        for i, count in self.counts.iteritems():
            assert count >= 0, "cannot score while in debt"
            prior_i = shared.betas[i] * shared.alpha
            score += gammaln(prior_i + count) - gammaln(prior_i)
        score += gammaln(shared.alpha) - gammaln(shared.alpha + self.total)
        return score

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def merge(self, shared, source):
        for i, count in source.counts.iteritems():
            self.add_repeated_value(shared, i, count)
        self.total += source.total

    def load(self, raw):
        self.counts = {}
        self.total = 0
        for i, count in raw['counts'].iteritems():
            if count:
                self.counts[int(i)] = int(count)
                self.total += count

    def dump(self):
        counts = {
            value: count
            for value, count in self.counts.iteritems()
            if count
        }
        return {'counts': counts}

    def protobuf_load(self, message):
        self.counts = {}
        self.total = 0
        for i, count in izip(message.keys, message.values):
            if count:
                self.counts[int(i)] = int(count)
                self.total += count

    def protobuf_dump(self, message):
        message.Clear()
        for i, count in self.counts.iteritems():
            if count:
                message.keys.append(i)
                message.values.append(count)


class Sampler(object):
    def init(self, shared, group=None):
        self.values = []
        post = []
        alpha = shared.alpha
        counts = {} if group is None else group.counts
        for value, beta in shared.betas.iteritems():
            self.values.append(value)
            post.append(beta * alpha + counts.get(value, 0))
        if shared.beta0 > 0:
            self.values.append(OTHER)
            post.append(shared.beta0 * alpha)
        self.probs = sample_dirichlet(post)

    def eval(self, shared):
        index = sample_discrete(self.probs)
        return self.values[index]


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
