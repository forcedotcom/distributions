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
Logistic Exponential Exponential 

lamb = Exp(lamb, lamb_hp)
mu = Exp(mu, mu_hp)

p_unscaled = Logistic(x | mu, lamb)
p = rescale(p_unscaled, p_min, p_max)

y = Bernoulli(p)

"""

from distributions.dbg.special import sqrt, log, pi, gammaln
from distributions.dbg.random import sample_chi2, sample_normal
from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'LogistcExpExp'
EXAMPLES = [
    {
        'shared': {'mu_hp': 1., 'lamb_hp': 1., 'p_min': 0.01, 'p_max': 0.99},
        'values': [(True, 1.0), (True, 1.5), (False, 2.0), (False, 3.0)]
    },
]

Value = tuple

def rev_logistic_scaled(x, mu, lamb, pmin, pmax):
    ratio = (x-mu) / lamb
    p_unscaled = 1.0 / (1.0 + np.exp(ratio))
    return p_unscaled * (pmax-pmin) + pmin

def log_exp_dist(x, lamb):
    if x < 0.0 : 
        return -np.inf
    if lamb <=0.0:
        return -np.inf

    return np.log(lamb)  - lamb*x


class Shared(SharedIoMixin):
    def __init__(self):
        self.mu_hp = None
        self.lamb_hp = None
        self.p_min
        self.p_max

    def plus_group(self, group):
        raise

    def load(self, raw):
        # self.mu = float(raw['mu'])
        # self.kappa = float(raw['kappa'])
        # self.sigmasq = float(raw['sigmasq'])
        # self.nu = float(raw['nu'])

    def dump(self):
        # return {
        #     'mu': self.mu,
        #     'kappa': self.kappa,
        #     'sigmasq': self.sigmasq,
        #     'nu': self.nu,
        # }

    def load_protobuf(self, message):
        # self.mu = float(message.mu)
        # self.kappa = float(message.kappa)
        # self.sigmasq = float(message.sigmasq)
        # self.nu = float(message.nu)

    def dump_protobuf(self, message):
        # message.Clear()
        # message.mu = self.mu
        # message.kappa = self.kappa
        # message.sigmasq = self.sigmasq
        # message.nu = self.nu


class Group(GroupIoMixin):
    def __init__(self):
        self.mu = None
        self.lamb = None

    def init(self, shared):
        """
        This just inits to a sane value

        """
        self.mu = 1.0
        self.lamb = 1.0

    def add_value(self, shared, value):
        """
        No-op in this case
        """
    def remove_value(self, shared, value):
        """
        No-op in this case
        """

    def merge(self, shared, source):
        """
        No-op in this case
        """

    def sample_params(self, shared):
        """
        Sample from the prior
        """
        self.mu = np.random.exp(1./shared['mu_hp'])
        self.lamb = np.random.exp(1./shared['lamb_hp'])

    def score_params(self, shared):
        """
        Score the parameters from the prior
        """
        score = log_exp_dist(self.mu, shared['mu_hp'])
        score += log_exp_dist(self.lamb, shared['lamb_hp'])
        return score

    def score_value(self, shared, value):
        p = rev_logistic_scaled(value[1], 
                                shared['mu_hp'], shared['lamb_hp'], 
                                shared['p_min'], shared['p_max'])
)
        if value[0]):
            return np.log(p)
        else:
            return np.log(1-p)

        return score

    def score_data(self, shared, data):
        return sum(self.score_value(shared, value) for value in data)

    def score_group(self, shared, data):
        return self.score_data(shared, data) + self.score_params(shared)

    def load(self, raw):
        raise 'todo'

    def dump(self):
        raise 'todo'

    def load_protobuf(self, message):
        raise 'todo'

    def dump_protobuf(self, message):
        raise 'todo'


# def sampler_create(shared, group=None):
#     """
#     Draw samples from the marginal posteriors of mu and sigmasq

#     \cite{murphy2007conjugate}, Eqs. 156 & 167
#     """
#     post = shared if group is None else shared.plus_group(group)
#     # Sample from the inverse-chi^2 using the transform from the chi^2
#     sigmasq_star = post.nu * post.sigmasq / sample_chi2(post.nu)
#     mu_star = sample_normal(post.mu, sqrt(sigmasq_star / post.kappa))
#     return (mu_star, sigmasq_star)


# def sampler_eval(shared, sampler):
#     mu, sigmasq = sampler
#     return sample_normal(mu, sqrt(sigmasq))


# def sample_value(shared, group):
#     sampler = sampler_create(shared, group)
#     return sampler_eval(shared, sampler)


# def sample_group(shared, size):
#     sampler = sampler_create(shared)
#     return [sampler_eval(shared, sampler) for _ in xrange(size)]
