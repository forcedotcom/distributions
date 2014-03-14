// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>
#include <random>
#include <distributions/random.hpp>

//----------------------------------------------------------------------------
// WARNING We use the pattern
//
//   auto sample = generate_foo(foo_distribution_t::param_type(param));
//
// rather than the pattern
//
//   generate_foo.param(param);
//   auto sample = generate_foo();
//
// because the param setter method
//
//   void foo_distribution_t::param(const foo_distribution_t::param_type &)
//
// was buggy in one version of std::chi_squared_distribution<double>.
//----------------------------------------------------------------------------

namespace std_wrapper
{

typedef distributions::rng_t rng_t;
extern rng_t global_rng;


namespace detail
{

typedef std::normal_distribution<double> normal_distribution_t;
extern normal_distribution_t generate_normal;

typedef std::chi_squared_distribution<double> chisq_distribution_t;
extern chisq_distribution_t generate_chisq;

typedef std::gamma_distribution<double> gamma_distribution_t;
extern gamma_distribution_t generate_gamma;

typedef std::poisson_distribution<int> poisson_distribution_t;
extern poisson_distribution_t generate_poisson;

extern std::uniform_real_distribution<double> generate_unif01;

} // namespace detail


inline void std_rng_seed(unsigned long s)
{
    global_rng.seed(s);
    detail::generate_normal.reset();
    detail::generate_chisq.reset();
    detail::generate_gamma.reset();
    detail::generate_poisson.reset();
    detail::generate_unif01.reset();
}

inline double std_random_normal(double mu, double sigmasq)
{
    typedef detail::normal_distribution_t::param_type param_type;
    return detail::generate_normal(global_rng, param_type(mu, sqrt(sigmasq)));
}

inline double std_random_chisq(double nu)
{
    typedef detail::chisq_distribution_t::param_type param_type;
    return detail::generate_chisq(global_rng, param_type(nu));
}

inline double std_random_gamma(double alpha, double beta)
{
    typedef detail::gamma_distribution_t::param_type param_type;
    return detail::generate_gamma(global_rng, param_type(alpha, beta));
}

inline int std_random_poisson(double mu)
{
    typedef detail::poisson_distribution_t::param_type param_type;
    return detail::generate_poisson(global_rng, param_type(mu));
}

template<class real_t>
inline int std_random_categorical(size_t D, const real_t * ps)
{
    double t = detail::generate_unif01(global_rng);
    for (size_t d = 0; d < D - 1; ++d) {
        t -= ps[d];
        if (t < 0) {
            return d;
        }
    }
    return D - 1;
}

template<class real_t>
inline void std_random_dirichlet(
        size_t D,
        const real_t * alphas,
        real_t * thetas)
{
    double total = 0.0;
    for (size_t d = 0; d < D; ++d) {
        total += thetas[d] = std_random_gamma(alphas[d], 1.0);
    }

    double scale = 1.0 / total;
    for (size_t d = 0; d < D; ++d) {
        thetas[d] *= scale;
    }
}

} // namespace std_wrapper
