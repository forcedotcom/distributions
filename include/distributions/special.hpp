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
#include <vector>
#include <cstring>
#include <iostream>
#include <limits>
#include <distributions/common.hpp>
#include <distributions/vendor/fmath.hpp>

#define M_PIf (3.14159265358979f)

namespace distributions {

template<class T> T sqr(const T & t) {
    return t * t;
}


// ---------------------------------------------------------------------------
// fast_log, fast_exp, fast_log_sum_exp, log_sum_exp

namespace detail {

/// Implements the ICSI fast log algorithm, v2.
class FastLog {
 public:
    explicit FastLog(int N);

    inline float log(float x) {
        // int intx = * reinterpret_cast<int *>(& x);
        int intx;
        memcpy(&intx, &x, 4);

        register const int exp = ((intx >> 23) & 255) - 127;
        register const int man = (intx & 0x7FFFFF) >> (23 - N_);

        // exponent plus lookup refinement
        return (static_cast<float>(exp) + table_[man]) * 0.69314718055994529f;
    }

 private:
    const int N_;
    std::vector<float> table_;
};

static FastLog GLOBAL_FAST_LOG_14(14);

}  // namespace detail

inline float eric_log(float x) {
    return detail::GLOBAL_FAST_LOG_14.log(x);
}

inline float fast_log(float x) {
    return eric_log(x);
    // return fmath::log(x);
}

inline float fast_exp(float x) {
    return fmath::exp(x);
}

inline float fast_log_sum_exp(float x, float y) {
    float min = x < y ? x : y;
    float max = x < y ? y : x;

    return max + fast_log(1.0f + fast_exp(min - max));
}

inline float log_sum_exp(float x, float y) {
    float min = x < y ? x : y;
    float max = x < y ? y : x;
    return max + logf(1.0f + expf(min - max));
}

// ---------------------------------------------------------------------------
// fast_lgamma, fast_log_beta, log_beta, log_binom, fast_log_binom

namespace detail {

extern const char LogTable256[256];
extern const float lgamma_approx_coeff5[];

}  // namespace detail

inline float fast_lgamma(float y) {
    // A piecewise fifth-order approximation of loggamma,
    // which bottoms out in libc gammaln for vals < 1.0
    // and throws an exception outside of the domain 2**32
    //
    // see loggamma.py for the code used to generate the coefficient table

    if (DIST_UNLIKELY(y < 2.5f or 4294967295.0f <= y)) {
        return lgammaf(y);
    }

    // adapted from:
    // http://www-graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
    float v = y;                // find int(log2(v)), where v > 0.0 && finite(v)
    int c;                      // 32-bit int c gets the result;
    int x = *(const int *) &v;  // or portably:  memcpy(&x, &v, sizeof x);

    c = x >> 23;

    if (c) {
        c -= 127;
    } else {  // subnormal, so recompute using mantissa: c = intlog2(x) - 149;
        register unsigned int t;  // temporary
        if ((t = x >> 16)) {
            c = detail::LogTable256[t] - 133;
        } else {
            c = (t = x >> 8)
              ? detail::LogTable256[t] - 141
              : detail::LogTable256[x] - 149;
        }
    }

    int pos = c *6;
    float a5 = detail::lgamma_approx_coeff5[pos];
    float a4 = detail::lgamma_approx_coeff5[pos + 1];
    float a3 = detail::lgamma_approx_coeff5[pos + 2];
    float a2 = detail::lgamma_approx_coeff5[pos + 3];
    float a1 = detail::lgamma_approx_coeff5[pos + 4];
    float a0 = detail::lgamma_approx_coeff5[pos + 5];

    double yprod = y;
    double sum = a0;
    sum += a1 * yprod;

    yprod *= y;
    sum += a2 * yprod;

    yprod *= y;
    sum += a3 * yprod;

    yprod *= y;
    sum += a4 * yprod;

    yprod *= y;
    sum += a5 * yprod;

    return sum;
}

inline float log_beta(float alpha, float beta) {
    if (DIST_UNLIKELY(alpha <= 0.f or beta <= 0.f)) {
        return - std::numeric_limits<float>::infinity();
    } else {
        return lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta);
    }
}

inline float fast_log_beta(float alpha, float beta) {
    if (DIST_UNLIKELY(alpha <= 0.f or beta <= 0.f)) {
        return - std::numeric_limits<float>::infinity();
    } else {
        return fast_lgamma(alpha)
             + fast_lgamma(beta)
             - fast_lgamma(alpha + beta);
    }
}

inline float log_binom(float N, float k) {
    return lgamma(N + 1) - (lgamma(k + 1) + lgamma(N - k + 1));
}

inline float fast_log_binom(float N, float k) {
    return fast_lgamma(N + 1) - (fast_lgamma(k + 1) + fast_lgamma(N - k + 1));
}

// ---------------------------------------------------------------------------
// fast_log_factorial

namespace detail {

extern const float log_factorial_table[64];

}  // namespace detail

inline float fast_log_factorial(const uint32_t & n) {
    if (n < 64) {
        return detail::log_factorial_table[n];
    } else {
        return fast_lgamma(n + 1);
    }
}


// ---------------------------------------------------------------------------
// fast_lgamma_nu

namespace detail {

extern const float lgamma_nu_func_approx_coeff3[];

inline float poly_eval_3(
        const float * __restrict__ coeff,
        float x) {
    // evaluate the polynomial with the indicated
    // coefficients
    float a0 = coeff[3];
    float a1 = coeff[2];
    float a2 = coeff[1];
    float a3 = coeff[0];

    return a0 + x*a1 + x*x*a2 + x*x*x*a3;
}

}  // namespace detail

inline float fast_lgamma_nu(float nu) {
    // Approximation of the sensitive, time-consuming
    // lgamma(nu / 2.0 + 0.5) - lgamma(nu/2.0)
    // function inside log student t

    // see loggamma.py:lstudent for coeff gen

    if (DIST_UNLIKELY(nu < 0.0625f or 4294967295.0f <= nu)) {
        return lgammaf(nu * 0.5f + 0.5f) - lgammaf(nu * 0.5f);
    }

    // adapted from:
    // http://www-graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
    float v = nu;               // find int(log2(v)), where v > 0.0 && finite(v)
    int c;                      // 32-bit int c gets the result;
    int x = *(const int *) &v;  // or portably:  memcpy(&x, &v, sizeof x);

    c = x >> 23;

    if (c) {
        c -= 127;
    } else {  // subnormal, so recompute using mantissa: c = intlog2(x) - 149;
        register unsigned int t;  // temporary
        if ((t = x >> 16)) {
            c = detail::LogTable256[t] - 133;
        } else {
            c = (t = x >> 8)
              ? detail::LogTable256[t] - 141
              : detail::LogTable256[x] - 149;
        }
    }

    int pos = ((c + 4) / 2) * 4;  // remember the POT range is 2
    return detail::poly_eval_3(detail::lgamma_nu_func_approx_coeff3 + pos, nu);
}

/**
 * http://en.wikipedia.org/wiki/Multivariate_gamma_function
 */
inline float lmultigamma(unsigned d, float a) {
    DIST_ASSERT1(d > 0, "zero dim lmultigamma");
    const float log_pi = 1.1447298858494002;
    const float term1 = 0.25 * static_cast<float>(d * (d - 1)) * log_pi;
    float term2 = 0.;
    for (int j = 1; j <= static_cast<int>(d); j++)
        term2 += fast_lgamma(a + 0.5 * static_cast<float>(1 - j));
    return term1 + term2;
}


// --------------------------------------------------------------------------
// misc

// Compute stirling numbers of first kind S(n,k), one row at a time
// return [log(S(n,0), ..., log(S(n,n))]
// http://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
template<class Alloc>
void get_log_stirling1_row(size_t n, std::vector<float, Alloc> & result);

inline std::vector<float> log_stirling1_row(size_t n) {
    std::vector<float> result;
    get_log_stirling1_row(n, result);
    return result;
}

}  // namespace distributions
