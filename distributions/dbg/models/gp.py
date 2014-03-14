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

from distributions.dbg.special import log, factorial, gammaln
from distributions.dbg.random import sample_gamma, sample_poisson
from distributions.mixins import ComponentModel, Serializable


class GammaPoisson(ComponentModel, Serializable):
    def __init__(self):
        self.alpha = None
        self.inv_beta = None

    def load(self, raw):
        self.alpha = float(raw['alpha'])
        self.inv_beta = float(raw['inv_beta'])

    def dump(self):
        return {
            'alpha': self.alpha,
            'inv_beta': self.inv_beta,
        }

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    class Group(object):
        def __init__(self):
            self.count = None
            self.sum = None
            self.log_prod = None

        def load(self, raw):
            self.count = int(raw['count'])
            self.sum = int(raw['sum'])
            self.log_prod = float(raw['log_prod'])

        def dump(self):
            return {
                'count': self.count,
                'sum': self.sum,
                'log_prod': self.log_prod,
            }

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, group):
        group.count = 0
        group.sum = 0
        group.log_prod = 0.

    def group_add_value(self, group, value):
        group.count += 1
        group.sum += int(value)
        group.log_prod += log(factorial(value))

    def group_remove_value(self, group, value):
        group.count -= 1
        group.sum -= int(value)
        group.log_prod -= log(factorial(value))

    def group_merge(self, destin, source):
        destin.count += source.count
        destin.sum += source.sum
        destin.log_prod += source.log_prod

    def plus_group(self, group):
        post = self.__class__()
        post.alpha = self.alpha + group.sum
        post.inv_beta = self.inv_beta + group.count
        return post

    #-------------------------------------------------------------------------
    # Sampling

    def sampler_create(self, group=None):
        post = self if group is None else self.plus_group(group)
        return sample_gamma(post.alpha, 1.0 / post.inv_beta)

    def sampler_eval(self, sampler):
        return sample_poisson(sampler)

    def sample_value(self, group):
        sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, size):
        sampler = self.sampler_create()
        return [self.sampler_eval(sampler) for _ in xrange(size)]

    #-------------------------------------------------------------------------
    # Scoring

    def score_value(self, group, value):
        post = self.plus_group(group)
        return gammaln(post.alpha + value) - gammaln(post.alpha) \
            + post.alpha * log(post.inv_beta) \
            - (post.alpha + value) * log(1. + post.inv_beta) \
            - log(factorial(value))

    def score_group(self, group):
        post = self.plus_group(group)
        return gammaln(post.alpha) - gammaln(self.alpha) \
            - post.alpha * log(post.inv_beta) \
            + self.alpha * log(self.inv_beta) \
            - group.log_prod

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alpha': 1., 'inv_beta': 1.},
            'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
        }
    ]


Model = GammaPoisson
