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


class DirichletDiscrete(ComponentModel, Serializable):
    def __init__(self):
        self.alphas = None

    @property
    def dim(self):
        return len(self.alphas)

    def load(self, raw):
        self.alphas = numpy.array(raw['alphas'], dtype=numpy.float)

    def dump(self):
        return {'alphas': self.alphas.tolist()}

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    class Group(object):
        def __init__(self):
            self.counts = None

        def load(self, raw):
            self.counts = numpy.array(raw['counts'], dtype=numpy.int)

        def dump(self):
            return {'counts': self.counts.tolist()}

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.counts = numpy.zeros(self.dim, dtype=numpy.int)

    def group_add_value(self, group, value):
        group.counts[value] += 1

    def group_remove_value(self, group, value):
        group.counts[value] -= 1

    def group_merge(self, destin, source):
        destin.counts += source.counts

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_create(self, group=None):
        if group is None:
            return sample_dirichlet(self.alphas)
        else:
            return sample_dirichlet(group.counts + self.alphas)

    def sampler_eval(self, sampler):
        return sample_discrete(sampler)

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
        \cite{wallach2009rethinking} Eqn 4.
        McCallum, et. al, 'Rething LDA: Why Priors Matter'
        """
        numer = group.counts[value] + self.alphas[value]
        denom = group.counts.sum() + self.alphas.sum()
        return log(numer / denom)

    def score_group(self, group):
        """
        \cite{jordan2001more} Eqn 22.
        Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """

        dim = self.dim
        a = self.alphas
        m = group.counts

        score = sum(gammaln(a[k] + m[k]) - gammaln(a[k]) for k in xrange(dim))
        score += gammaln(a.sum())
        score -= gammaln(a.sum() + m.sum())
        return score

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alphas': [1.0, 4.0]},
            'values': [0, 1, 1, 1, 1, 0, 1],
        },
        {
            'model': {'alphas': [0.5] * 4},
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
        #{
        #    'model': {'alphas': [0.5] * 256},
        #    'values': [0, 1, 3, 7, 15, 31, 63, 127, 255],
        #},
    ]


Model = DirichletDiscrete
