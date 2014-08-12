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

cimport numpy as np
import numpy as np

### Helpers to convert Eigen <-> numpy

cdef VectorXf to_eigen_vecf(np.ndarray x):
    cdef VectorXf v = VectorXf(x.shape[0])
    for i, e in enumerate(x):
        v[i] = float(e)
    return v

cdef MatrixXf to_eigen_matf(np.ndarray x):
    cdef MatrixXf m = MatrixXf(x.shape[0], x.shape[1])
    for i, a in enumerate(x):
        for j, b in enumerate(a):
            _h.float_op_set2(m, i, j, b)
    return m

cdef np.ndarray to_np_1darray(const VectorXf &x):
    cdef np.ndarray v = np.zeros(x.size())
    for i in xrange(x.size()):
        v[i] = x[i]
    return v

cdef np.ndarray to_np_2darray(const MatrixXf &x):
    cdef np.ndarray m = np.zeros((x.rows(), x.cols()))
    for i in xrange(x.rows()):
        for j in xrange(x.cols()):
            m[i, j] = _h.float_op_get2(x, i, j)
    return m
