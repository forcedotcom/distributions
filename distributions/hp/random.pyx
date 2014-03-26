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
cimport numpy

from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng

cdef extern from 'rng_cc.hpp' namespace 'std_wrapper':
    cdef void std_rng_seed(rng_t & rng, unsigned long seed)
    cdef double std_random_normal(rng_t & rng, double mu, double sigmasq)
    cdef double std_random_chisq(rng_t & rng, double nu)
    cdef double std_random_gamma(rng_t & rng, double alpha, double beta)
    cdef int std_random_poisson(rng_t & rng, double mu)
    cdef int std_random_categorical(rng_t & rng, size_t dim, double ps[])
    cdef void std_random_dirichlet(
            rng_t & rng,
            size_t dim,
            double alphas[],
            double thetas[])


cpdef int random():
    return get_rng()[0].sample()


cpdef seed(unsigned long s):
    std_rng_seed(get_rng()[0], s)


cpdef double sample_normal(double mu, double sigmasq):
    return std_random_normal(get_rng()[0], mu, sigmasq)


cpdef double sample_chisq(double nu):
    return std_random_chisq(get_rng()[0], nu)


cpdef sample_gamma(double a, double b):
    return std_random_gamma(get_rng()[0], a, b)


cpdef sample_poisson(double mu):
    return std_random_poisson(get_rng()[0], mu)


cdef sample_discrete(size_t dim, double ps[]):
    return std_random_categorical(get_rng()[0], dim, ps)


cdef sample_dirichlet(size_t dim, double alphas[], double thetas[]):
    std_random_dirichlet(get_rng()[0], dim, alphas, thetas)


#def sample_discrete(numpy.ndarray[double, ndim=1] ps):
#    cdef int dim = ps.shape[0]
#    return std_random_categorical(dim, &ps[0])
#
#
#def sample_dirichlet(
#        numpy.ndarray[double, ndim=1] alphas,
#        numpy.ndarray[double, ndim=1] thetas):
#    cdef int dim = alphas.shape[0]
#    std_random_dirichlet(dim, &alphas[0], &thetas[0])
