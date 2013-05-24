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

import distributions.summaries
from nose.tools import ok_, assert_almost_equal


def test_summarize_continuous():
    x = [0.]
    summary = distributions.summaries.summarize_continuous(x)
    ok_(summary['n'] == 1)

    x = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    summary = distributions.summaries.summarize_continuous(x)
    ok_(summary['n'] == 10)
    assert_almost_equal(summary['sum'], 45.)
    assert_almost_equal(summary['mean'], 4.5)
    assert_almost_equal(summary['median'], 4.5)


def test_summarize_categorical():
    x = [0]
    summary = distributions.summaries.summarize_categorical(x)
    ok_(summary['n'] == 1)

    x = [0, 0, 1, 2, 2, 3, 4, 4, 4, 6]
    summary = distributions.summaries.summarize_categorical(x)
    ok_(summary['n'] == 10)
    ok_(summary['dim'] == 7)
    ok_(summary['mode'] == 4)
