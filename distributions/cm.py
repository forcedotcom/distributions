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

from distributions.conjugate import (
    bb,
    dd,
    cydd,
    dpm,
    cydpm,
    gp,
    cygp,
    ngig,
    cynich,
    nich,
    niw,
    )

from distributions import (
    crp,
    pyp
    )

NAMES = {
    'asymmetricdirichletdiscrete': cydd,
    'bb': bb,
    'betabernoulli': bb,
    'crp': crp,
    'dd': cydd,
    'cydd': cydd,
    'pydd': dd,
    'dirichletprocessmultinomial': dpm,
    'dpm': cydpm,
    'cydpm': cydpm,
    'pydpm': dpm,
    'gp': cygp,
    'cygp': cygp,
    'pygp': gp,
    'gammapoisson': cygp,
    'ngig': ngig,
    'nich': cynich,
    'cynich': cynich,
    'pynich': nich,
    'niw': niw,
    'normalinversechisq': cynich,
    'normalinversewishart': niw,
    'pyp': pyp,
    }


class ComponentModel:
    def __init__(self, name, p=None, hp=None, ss=None):
        self.mod = NAMES[name.lower()]
        self.hp = self.mod.create_hp(hp, p)
        self.ss = self.mod.create_ss(ss, p)

    def add_data(self, y):
        return self.mod.add_data(self.ss, y)

    def remove_data(self, y):
        return self.mod.remove_data(self.ss, y)

    def sample_data(self, N=1):
        if N == 1:
            return self.mod.sample_data(self.hp, self.ss)
        else:
            return [self.mod.sample_data(self.hp, self.ss) for _ in range(N)]

    def sample_post(self):
        return self.mod.sample_post(self.hp, self.ss)

    def pred_prob(self, y):
        return self.mod.pred_prob(self.hp, self.ss, y)

    def data_prob(self):
        return self.mod.data_prob(self.hp, self.ss)

    def dump_ss(self):
        return self.mod.dump_ss(self.ss)

    def dump_hp(self):
        return self.mod.dump_hp(self.hp)

    def realize_hp(self):
        '''
        This only applies to partial hp representations
        '''
        if hasattr(self.mod, 'realize_hp'):
            self.mod.realize_hp(self.hp)

    def generate_post(self):
        return self.mod.generate_post(self.hp, self.ss)
