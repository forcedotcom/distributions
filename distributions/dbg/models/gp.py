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
from distributions.mixins import (
    ComponentModel,
    Serializable,
    ProtobufSerializable,
)

NAME = 'GammaPoisson'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'inv_beta': 1.},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    }
]
Value = int


class Shared(
        ComponentModel,
        Serializable,
        ProtobufSerializable):

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

    def load_protobuf(self, message):
        self.alpha = float(message.alpha)
        self.inv_beta = float(message.inv_beta)

    def dump_protobuf(self, message):
        message.Clear()
        message.alpha = self.alpha
        message.inv_beta = self.inv_beta

    #-------------------------------------------------------------------------
    # Mutation

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


class Group(ProtobufSerializable):
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

    def load_protobuf(self, message):
        self.count = int(message.count)
        self.sum = int(message.sum)
        self.log_prod = float(message.log_prod)

    def dump_protobuf(self, message):
        message.count = self.count
        message.sum = self.sum
        message.log_prod = self.log_prod

    def init(self, model):
        self.count = 0
        self.sum = 0
        self.log_prod = 0.

    def add_value(self, model, value):
        self.count += 1
        self.sum += int(value)
        self.log_prod += log(factorial(value))

    def remove_value(self, model, value):
        self.count -= 1
        self.sum -= int(value)
        self.log_prod -= log(factorial(value))

    def merge(self, model, source):
        self.count += source.count
        self.sum += source.sum
        self.log_prod += source.log_prod


# temporary refactoring kludge
Shared.Group = Group


def score_value(model, group, value):
    post = model.plus_group(group)
    return gammaln(post.alpha + value) - gammaln(post.alpha) \
        + post.alpha * log(post.inv_beta) \
        - (post.alpha + value) * log(1. + post.inv_beta) \
        - log(factorial(value))


def score_group(model, group):
    post = model.plus_group(group)
    return gammaln(post.alpha) - gammaln(model.alpha) \
        - post.alpha * log(post.inv_beta) \
        + model.alpha * log(model.inv_beta) \
        - group.log_prod
