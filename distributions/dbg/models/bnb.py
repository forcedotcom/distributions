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
Cook, John D. "Notes on the negative binomial distribution."
Unknown, October 28 (2009): 2009.

Following http://www.johndcook.com/negative_binomial.pdf
The negative binomial (NB) gives the probability of seeing x
failures before the rth success given a success probability
p:
    p(x | p, r) \propto p ^ r * (1 - p) ^ x
For a given r and p, the NB has mean:
    mu = r (1 - p) / p
and variance:
    sigmasq = mu + (1 / r) * mu ** 2
"""
from distributions.dbg.special import gammaln
from distributions.dbg.random import sample_beta, sample_negative_binomial
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'BetaNegativeBinomial'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'beta': 1., 'r': 1},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    },
]


Value = int


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.r = None

    def plus_group(self, group):
        post = self.__class__()
        post.alpha = self.alpha + self.r * group.count
        post.beta = self.beta + group.sum
        post.r = self.r
        return post

    def load(self, raw):
        self.alpha = float(raw['alpha'])
        self.beta = float(raw['beta'])
        self.r = int(raw['r'])

    def dump(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r': self.r,
        }

    def protobuf_load(self, message):
        self.alpha = float(message.alpha)
        self.beta = float(message.beta)
        self.r = int(message.r)

    def protobuf_dump(self, message):
        message.Clear()
        message.alpha = self.alpha
        message.beta = self.beta
        message.r = self.r


class Group(GroupIoMixin):
    def __init__(self):
        self.count = None
        self.sum = None

    def init(self, shared):
        self.count = 0
        self.sum = 0

    def add_value(self, shared, value):
        self.count += 1
        self.sum += int(value)

    def add_repeated_value(self, shared, value, count):
        self.count += count
        self.sum += count * int(value)

    def remove_value(self, shared, value):
        self.count -= 1
        self.sum -= int(value)

    def merge(self, shared, source):
        self.count += source.count
        self.sum += source.sum

    def score_value(self, shared, value):
        post = shared.plus_group(self)
        alpha = post.alpha + shared.r
        beta = post.beta + value
        score = gammaln(post.alpha + post.beta)
        score -= gammaln(alpha + beta)
        score += gammaln(alpha) - gammaln(post.alpha)
        score += gammaln(beta) - gammaln(post.beta)
        return score

    def score_data(self, shared):
        post = shared.plus_group(self)
        score = gammaln(shared.alpha + shared.beta)
        score -= gammaln(post.alpha + post.beta)
        score += gammaln(post.alpha) - gammaln(shared.alpha)
        score += gammaln(post.beta) - gammaln(shared.beta)
        return score

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def dump(self):
        return {
            'count': self.count,
            'sum': self.sum,
        }

    def load(self, raw):
        self.count = int(raw['count'])
        self.sum = int(raw['sum'])

    def protobuf_load(self, message):
        self.count = int(message.count)
        self.sum = int(message.sum)

    def protobuf_dump(self, message):
        message.count = self.count
        message.sum = self.sum


class Sampler(object):
    def init(self, shared, group=None):
        post = shared if group is None else shared.plus_group(group)
        self.p = sample_beta(post.alpha, post.beta)

    def eval(self, shared):
        return sample_negative_binomial(self.p, shared.r)


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
