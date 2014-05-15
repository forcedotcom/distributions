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

import numpy
from itertools import izip
from distributions.dbg.special import log, gammaln
from distributions.dbg.random import sample_discrete, sample_dirichlet
from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
            'gamma': 0.5,
            'alpha': 0.5,
            'betas': {  # beta0 must be zero for unit tests
                '0': 0.25,
                '1': 0.5,
                '2': 0.25,
            },
        },
        'values': [0, 1, 0, 2, 0, 1, 0],
    },
]
OTHER = -1
Value = int


class Shared(SharedIoMixin):
    def __init__(self):
        self.gamma = None
        self.alpha = None
        self.betas = None
        self.beta0 = None

    def _load_beta0(self):
        self.beta0 = 1 - self.betas.sum()
        if not (0 <= self.betas.min() and self.betas.max() <= 1):
            raise ValueError('betas out of bounds: {}'.format(self.betas))
        if not (0 <= self.beta0 and self.beta0 <= 1):
            raise ValueError('beta0 out of bounds: {}'.format(self.beta0))

    def load(self, raw):
        self.gamma = float(raw['gamma'])
        self.alpha = float(raw['alpha'])
        raw_betas = raw['betas']
        betas = [raw_betas[str(i)] for i in xrange(len(raw_betas))]
        self.betas = numpy.array(betas, dtype=numpy.float)  # dense
        self._load_beta0()

    def dump(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'betas': {str(i): beta for i, beta in enumerate(self.betas)},
        }

    def load_protobuf(self, message):
        self.gamma = float(message.gamma)
        self.alpha = float(message.alpha)
        self.betas = numpy.array(message.betas, dtype=numpy.float)
        self._load_beta0()

    def dump_protobuf(self, message):
        message.Clear()
        message.gamma = self.gamma
        message.alpha = self.alpha
        for beta in self.betas:
            message.betas.append(beta)


class Group(GroupIoMixin):
    def __init__(self):
        self.counts = None
        self.total = None

    def init(self, model):
        self.counts = {}  # sparse
        self.total = 0

    def add_value(self, model, value):
        assert value != OTHER, 'tried to add OTHER to suffstats'
        try:
            self.counts[value] += 1
        except KeyError:
            self.counts[value] = 1
        self.total += 1

    def remove_value(self, model, value):
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

    def merge(self, model, source):
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
            str(i): count
            for i, count in self.counts.iteritems()
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


def sampler_create(shared, group=None):
    probs = (shared.betas * shared.alpha).tolist()
    if group is not None:
        for i, count in group.counts.iteritems():
            probs[i] += count
    probs.append(shared.beta0 * shared.alpha)
    return sample_dirichlet(probs)


def sampler_eval(shared, sampler):
    index = sample_discrete(sampler)
    if index == len(shared.betas):
        return OTHER
    else:
        return index


def sample_value(shared, group):
    sampler = sampler_create(shared, group)
    return sampler_eval(shared, sampler)


def sample_group(shared, size):
    sampler = sampler_create(shared)
    return [sampler_eval(shared, sampler) for _ in xrange(size)]
