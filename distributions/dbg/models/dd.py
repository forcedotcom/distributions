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
from distributions.dbg.special import log, gammaln
from distributions.dbg.random import sample_discrete, sample_dirichlet
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'DirichletDiscrete'
EXAMPLES = [
    {
        'shared': {'alphas': [0.5] * 4},
        'values': [0, 1, 0, 2, 0, 1, 0],
    },
    {
        'shared': {'alphas': [1.0, 4.0]},
        'values': [0, 1, 1, 1, 1, 0, 1],
    },
    {
        'shared': {'alphas': [2.0 / n for n in xrange(1, 21)]},
        'values': range(20),
    },
]
Value = int


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.alphas = None

    @property
    def dim(self):
        return len(self.alphas)

    def load(self, raw):
        self.alphas = numpy.array(raw['alphas'], dtype=numpy.float)

    def dump(self):
        return {'alphas': self.alphas.tolist()}

    def protobuf_load(self, message):
        self.alphas = numpy.array(message.alphas, dtype=numpy.float)

    def protobuf_dump(self, message):
        message.Clear()
        for alpha in self.alphas:
            message.alphas.append(alpha)


class Group(GroupIoMixin):
    def __init__(self):
        self.counts = None

    def init(self, shared):
        self.counts = numpy.zeros(shared.dim, dtype=numpy.int)

    def add_value(self, shared, value):
        self.counts[value] += 1

    def add_repeated_value(self, shared, value, count):
        self.counts[value] += count

    def remove_value(self, shared, value):
        self.counts[value] -= 1

    def merge(self, shared, source):
        self.counts += source.counts

    def score_value(self, shared, value):
        """
        \cite{wallach2009rethinking} Eqn 4.
        McCallum, et. al, 'Rething LDA: Why Priors Matter'
        """
        numer = self.counts[value] + shared.alphas[value]
        denom = self.counts.sum() + shared.alphas.sum()
        return log(numer / denom)

    def score_data(self, shared):
        """
        \cite{jordan2001more} Eqn 22.
        Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """

        dim = shared.dim
        a = shared.alphas
        m = self.counts

        score = sum(gammaln(a[k] + m[k]) - gammaln(a[k]) for k in xrange(dim))
        score += gammaln(a.sum())
        score -= gammaln(a.sum() + m.sum())
        return score

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        self.counts = numpy.array(raw['counts'], dtype=numpy.int)

    def dump(self):
        return {'counts': self.counts.tolist()}

    def protobuf_load(self, message):
        self.counts = numpy.array(message.counts, dtype=numpy.int)

    def protobuf_dump(self, message):
        message.Clear()
        for count in self.counts:
            message.counts.append(count)


class Sampler(object):
    def init(self, shared, group=None):
        if group is None:
            self.ps = sample_dirichlet(shared.alphas)
        else:
            self.ps = sample_dirichlet(group.counts + shared.alphas)

    def eval(self, shared):
        return sample_discrete(self.ps)


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
