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
from nose.tools import assert_equal
from distributions.tests.util import require_cython, assert_close


def test_log_stirling1_row():
    require_cython()
    from distributions.lp.special import log_stirling1_row
    MAX_N = 128

    rows = [[1]]
    for n in range(1, MAX_N + 1):
        prev = rows[-1]
        middle = [(n - 1) * prev[k] + prev[k - 1] for k in range(1, n)]
        row = [0] + middle + [1]
        rows.append(row)

    for n in range(1, MAX_N + 1):
        print 'Row {}:'.format(n),
        row_py = numpy.log(numpy.array(rows[n][1:], dtype=numpy.double))
        row_cpp = log_stirling1_row(n)[1:]
        assert_equal(len(row_py), len(row_cpp))

        # only the slopes need to be accurate
        #print 0,
        #assert_close(row_py[0], row_cpp[0])
        #print len(row_py)
        #assert_close(row_py[-1], row_cpp[-1])

        diff_py = numpy.diff(row_py)
        diff_cpp = numpy.diff(row_cpp)
        for k_minus_1, (dx_py, dx_cpp) in enumerate(zip(diff_py, diff_cpp)):
            k = k_minus_1 + 1
            print '%d-%d' % (k, k + 1),
            assert_close(dx_py, dx_cpp, tol=0.5)
        print
