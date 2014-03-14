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
from distributions.mixins import ComponentModel, Serializable


OTHER = -1


class DirichletProcessDiscrete(ComponentModel, Serializable):
    def __init__(self):
        self.gamma = None
        self.alpha = None
        self.betas = None
        self.beta0 = None

    def load(self, raw):
        self.gamma = float(raw['gamma'])
        self.alpha = float(raw['alpha'])
        raw_betas = raw['betas']
        betas = [raw_betas[str(i)] for i in xrange(len(raw_betas))]
        self.betas = numpy.array(betas, dtype=numpy.float)  # dense
        self.beta0 = 1 - self.betas.sum()
        if not (0 <= self.betas.min() and self.betas.max() <= 1):
            raise ValueError('betas out of bounds: {}'.format(self.betas))
        if not (0 <= self.beta0 and self.beta0 <= 1):
            raise ValueError('beta0 out of bounds: {}'.format(self.beta0))

    def dump(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'betas': {str(i): beta for i, beta in enumerate(self.betas)},
        }

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    class Group(object):
        def __init__(self):
            self.counts = None
            self.total = None

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

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.counts = {}  # sparse
        group.total = 0

    def group_add_value(self, group, value):
        assert value != OTHER, 'tried to add OTHER to suffstats'
        try:
            group.counts[value] += 1
        except KeyError:
            group.counts[value] = 1
        group.total += 1

    def group_remove_value(self, group, value):
        assert value != OTHER, 'tried to remove OTHER to suffstats'
        new_count = group.counts[value] - 1
        if new_count == 0:
            del group.counts[value]
        else:
            group.counts[value] = new_count
        group.total -= 1

    def group_merge(self, destin, source):
        for i, count in source.counts.iteritems():
            destin.counts[i] = destin.counts.get(i, 0) + count
        destin.total += source.total

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_create(self, group=None):
        probs = (self.betas * self.alpha).tolist()
        if group is not None:
            for i, count in group.counts.iteritems():
                probs[i] += count
        probs.append(self.beta0 * self.alpha)
        return sample_dirichlet(probs)

    def sampler_eval(self, sampler):
        index = sample_discrete(sampler)
        if index == len(self.betas):
            return OTHER
        else:
            return index

    def sample_value(self, group):
        sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, size):
        sampler = self.sampler_create()
        return [self.sampler_eval(sampler) for _ in xrange(size)]

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, group, value):
        """
        Adapted from dd.py, which was adapted from:
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        denom = self.alpha + group.total
        if value == OTHER:
            numer = self.beta0 * self.alpha
        else:
            numer = self.betas[value] * self.alpha + group.counts.get(value, 0)
        return log(numer / denom)

    def score_group(self, group):
        assert len(self.betas), 'betas is empty'
        """
        See doc/dpd.pdf Equation (3)
        """
        score = 0.
        for i, count in group.counts.iteritems():
            prior_i = self.betas[i] * self.alpha
            score += gammaln(prior_i + count) - gammaln(prior_i)
        score += gammaln(self.alpha) - gammaln(self.alpha + group.total)
        return score

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {
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


Model = DirichletProcessDiscrete
