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

from itertools import izip
from distributions.dbg.special import log, gammaln
from distributions.dbg.random import sample_discrete, sample_dirichlet
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
            'gamma': 0.5,
            'alpha': 0.5,
            'betas': {  # beta0 must be zero for unit tests
                0: 0.25,
                7: 0.5,
                8: 0.25,
            },
            'counts': {
                0: 1,
                7: 2,
                8: 4,
            }
        },
        'values': [0, 7, 0, 8, 0, 7, 0],
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
        self.beta0 = 1 - sum(self.betas.itervalues())
        min_beta = min(self.betas.itervalues())
        max_beta = max(self.betas.itervalues())
        if not (0 <= min_beta and max_beta <= 1):
            raise ValueError('betas out of bounds: {}'.format(self.betas))
        if not (0 <= self.beta0 and self.beta0 <= 1):
            raise ValueError('beta0 out of bounds: {}'.format(self.beta0))

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

    def load_protobuf(self, message):
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

    def dump_protobuf(self, message):
        message.Clear()
        message.gamma = self.gamma
        message.alpha = self.alpha
        for value, beta in self.betas.iteritems():
            message.values.append(value)
            message.betas.append(beta)
            message.counts.append(self.counts[value])

    def add_value(self, value):
        if value in self.counts:
            self.counts[value] += 1
            return False
        else:
            self.counts[value] = 1
            return True

    def remove_value(self, value):
        count = self.counts[value] - 1
        if count:
            self.counts[value] = count
            return False
        else:
            del self.counts[value]
            return True


class Group(GroupIoMixin):
    def __init__(self):
        self.counts = None
        self.total = None

    def init(self, shared):
        self.counts = {}  # sparse
        self.total = 0

    def add_value(self, shared, value):
        assert value != OTHER, 'tried to add OTHER to suffstats'
        try:
            self.counts[value] += 1
        except KeyError:
            self.counts[value] = 1
        self.total += 1

    def remove_value(self, shared, value):
        assert value != OTHER, 'tried to remove OTHER to suffstats'
        new_count = self.counts[value] - 1
        if new_count == 0:
            del self.counts[value]
        else:
            self.counts[value] = new_count
        self.total -= 1

    def score_value(self, shared, value):
        """
        Adapted from dd.py, which was adapted from:
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        denom = shared.alpha + self.total
        if value == OTHER:
            numer = shared.beta0 * shared.alpha
        else:
            numer = (
                shared.betas[value] * shared.alpha +
                self.counts.get(value, 0))
        return log(numer / denom)

    def score_data(self, shared):
        assert len(shared.betas), 'betas is empty'
        """
        See doc/dpd.pdf Equation (3)
        """
        score = 0.
        for i, count in self.counts.iteritems():
            prior_i = shared.betas[i] * shared.alpha
            score += gammaln(prior_i + count) - gammaln(prior_i)
        score += gammaln(shared.alpha) - gammaln(shared.alpha + self.total)
        return score

    def merge(self, shared, source):
        for i, count in source.counts.iteritems():
            self.counts[i] = self.counts.get(i, 0) + count
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

    def load_protobuf(self, message):
        self.counts = {}
        self.total = 0
        for i, count in izip(message.keys, message.values):
            if count:
                self.counts[int(i)] = int(count)
                self.total += count

    def dump_protobuf(self, message):
        message.Clear()
        for i, count in self.counts.iteritems():
            if count:
                message.keys.append(i)
                message.values.append(count)


class Sampler(object):
    def init(self, shared, group):
        self.values = []
        post = []
        alpha = shared.alpha
        for value, beta in shared.betas.iteritems():
            self.values.append(value)
            post.append(beta * alpha + group.counts.get(value, 0))
        if shared.beta0 > 0:
            self.values.append(OTHER)
            post.append(shared.beta0 * alpha)
        self.probs = sample_dirichlet(post)

    def eval(self, shared):
        index = sample_discrete(self.probs)
        return self.values[index]


def sampler_create(shared, group=None):
    if group is None:
        group = Group()
        group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return sampler


def sampler_eval(shared, sampler):
    return sampler.eval(shared)


def sample_value(shared, group):
    sampler = Sampler()
    sampler.init(shared, group)
    return sampler.eval(shared)


def sample_group(shared, size):
    sampler = sampler_create(shared)
    return [sampler.eval(shared) for _ in xrange(size)]
