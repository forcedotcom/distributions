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

from libc.math cimport sqrt, lgamma


cdef extern from 'std_wrapper.hpp' namespace 'std_wrapper':
    cdef void std_rng_seed(unsigned long seed)
    cdef double std_random_normal(double mu, double sigmasq)
    cdef double std_random_chisq(double nu)
    cdef double std_random_gamma(double alpha, double beta)
    cdef int std_random_poisson(double mu)
    cdef int std_random_categorical(size_t dim, double ps[])
    cdef void std_random_dirichlet(
            size_t dim,
            double alphas[],
            double thetas[])


cpdef seed(unsigned long s):
    std_rng_seed(s)


cpdef double normal_draw(double mu, double sigmasq):
    return std_random_normal(mu, sigmasq)


cpdef double chisq_draw(double nu):
    return std_random_chisq(nu)


cpdef gamma_draw(double a, double b):
    return std_random_gamma(a, b)


cpdef poisson_draw(double mu):
    return std_random_poisson(mu)


cdef categorical_draw(size_t K, double p[]):
    return std_random_categorical(K, p)


cdef dirichlet_draw(size_t K, double alpha[], double theta[]):
    std_random_dirichlet(K, alpha, theta)



cdef double logfactorial(unsigned n):
    return lgamma(n + 1)


cdef double gammaln(double x):
    return lgamma(x)
