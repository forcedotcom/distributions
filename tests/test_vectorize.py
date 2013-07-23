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


import numpy
from nose.tools import assert_equal, assert_almost_equal, assert_less
from distributions import ComponentModel
from util import check_cm

DPS = 100
COMPS = 10

MODELS = [
    'cynich',
    'cydd',
    'cydpm',
    'cygp',
    ]


def test_vectorize():
    for name in MODELS:
        check_cm(name)
        cm0 = ComponentModel(name)
        cm0.realize_hp()
        hp0 = cm0.dump_hp()
        cms = [ComponentModel(name, hp=hp0) for _ in range(COMPS)]
        for cm in cms:
            dps = [cm.sample_data() for _ in range(DPS)]
            for dp in dps:
                cm.add_data(dp)

        mod = cms[0].mod
        hp = cms[0].hp
        ss = [cm.ss for cm in cms]
        for cm in cms:
            y = cm.sample_data()
            scores = numpy.zeros(COMPS)
            mod.add_pred_probs(hp, ss, y, scores)
            for cm, score in zip(cms, scores):
                assert_almost_equal(score, cm.pred_prob(y))
