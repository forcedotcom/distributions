# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from distributions.names import CANONICAL

from distributions.basic import (
    Bernoulli,
    Discrete,
    InverseGaussian,
    MultivariateNormal,
    Normal,
    Poisson,
    )

# Set up the names so that if a component model name is passed in,
# the correct likelihood distribtuion is created.
NAMES = {
    'bernoulli': Bernoulli,
    'discrete': Discrete,
    'inversegaussian': InverseGaussian,
    'multivariatenormal': MultivariateNormal,
    'normal': Normal,
    'poisson': Poisson,
    }

CONJUGATE_NAMES = {
    'dd': 'discrete',
    'bb': 'bernoulli',
    'gp': 'poisson',
    'ngig': 'inversegaussian',
    'nich': 'normal',
    'niw': 'multivariatenormal',
    'dpm': 'discrete'
    }


class BasicDistribution:
    def __init__(self, name, pm=None):
        name = name.lower()
        name = CANONICAL.get(name, name)
        name = CONJUGATE_NAMES.get(name, name)
        self.mod = NAMES[name]
        self.pm = self.mod.create_p(pm)

    def sample_data(self, N=1):
        return self.mod.sample_data(self.pm, N)

    def data_prob(self, y):
        return self.mod.data_prob(self.pm, y)

    def dump_p(self):
        return self.mod.dump_p(self.pm)
