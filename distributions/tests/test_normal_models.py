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

import numpy as np

from nose import SkipTest
from distributions.tests.util import assert_close


def _test_normals(nich, niw):
    mu = np.array([30.0])
    kappa = 0.3
    psi = np.array([[2.]])
    nu = 3

    # make the NIW case
    niw_shared = niw.Shared()
    niw_shared.load({'mu': mu, 'kappa': kappa, 'psi': psi, 'nu': nu})
    niw_group = niw.Group()
    niw_group.init(niw_shared)

    # make the NIX case
    nix_shared = nich.Shared()
    nix_shared.load({
        'mu': mu[0],
        'kappa': kappa,
        'sigmasq': psi[0, 0] / nu,
        'nu': nu
    })
    nix_group = nich.Group()
    nix_group.init(nix_shared)

    data = np.array([4., 54., 3., -12., 7., 10.])
    for d in data:
        niw_group.add_value(niw_shared, np.array([d]))
        nix_group.add_value(nix_shared, d)

    # check marginals
    assert_close(niw_group.score_data(niw_shared),
                 nix_group.score_data(nix_shared))

    # remove and check
    niw_group.remove_value(niw_shared, np.array([data[1]]))
    nix_group.remove_value(nix_shared, np.array([data[1]]))

    assert_close(niw_group.score_data(niw_shared),
                 nix_group.score_data(nix_shared))

    niw_group.remove_value(niw_shared, np.array([data[3]]))
    nix_group.remove_value(nix_shared, np.array([data[3]]))

    assert_close(niw_group.score_data(niw_shared),
                 nix_group.score_data(nix_shared))

    # check posterior predictive
    values = np.array([32., -0.1])

    for value in values:
        assert_close(niw_group.score_value(niw_shared, np.array([value])),
                     nix_group.score_value(nix_shared, value))


def test_normals_dbg():
    try:
        from distributions.dbg.models import nich, niw
    except ImportError:
        raise SkipTest("no dbg.{nich,niw}")
    _test_normals(nich, niw)


def test_normals_lp():
    try:
        from distributions.lp.models import nich, niw
    except ImportError:
        raise SkipTest("no dbg.{nich,niw}")
    _test_normals(nich, niw)
