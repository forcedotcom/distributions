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

from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions._eigen_h cimport VectorXf, MatrixXf

ctypedef VectorXf Value

cdef extern from "distributions/models/niw.hpp" namespace "distributions::NormalInverseWishart<-1>":
    cppclass Shared:
        VectorXf mu
        float kappa
        MatrixXf psi
        float nu

    cppclass Group:
        int count
        VectorXf sum_x
        MatrixXf sum_xxT

        void init (Shared &, rng_t &) nogil except +
        void add_value (Shared &, Value &, rng_t &) nogil except +
        void add_repeated_value (Shared &, Value &, int &, rng_t &) nogil except +
        void remove_value (Shared &, Value &, rng_t &) nogil except +
        void merge (Shared &, Group &, rng_t &) nogil except +
        float score_value (Shared &, Value &, rng_t &) nogil except +
        float score_data (Shared &, rng_t &) nogil except +
        Value sample_value (Shared &, rng_t &) nogil except +

    cppclass Sampler:
        void init (Shared &, Group &, rng_t &) nogil except +
        Value eval (Shared &, rng_t &) nogil except +
