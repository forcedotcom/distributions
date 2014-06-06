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

#include <utility>
#include <random>
#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/random_fwd.hpp>

namespace distributions
{

inline int sample_int (rng_t & rng, int low, int high)
{
    std::uniform_int_distribution<> sampler(low, high);
    return sampler(rng);
}

inline float sample_unif01 (rng_t & rng)
{
    std::uniform_real_distribution<float> sampler(0.0, 1.0);
    return sampler(rng);
}

inline bool sample_bernoulli (rng_t & rng, float p)
{
    std::uniform_real_distribution<float> sampler(0.0, 1.0);
    return sampler(rng) < p;
}

inline float sample_std_normal (rng_t & rng)
{
    std::normal_distribution<float> sampler(0.f, 1.f);
    return sampler(rng);
}

inline float sample_normal (rng_t & rng, float mean, float variance)
{
    float stddev = sqrtf(variance);
    std::normal_distribution<float> sampler(mean, stddev);
    return sampler(rng);
}

inline float sample_chisq (rng_t & rng, float nu)
{
    // HACK <float> appears to be broken in libstdc++ 4.6
    //typedef std::chi_squared_distribution<float> chi_squared_distribution_t;
    typedef std::chi_squared_distribution<double> chi_squared_distribution_t;

    chi_squared_distribution_t sampler(nu);
    return sampler(rng);
}

inline int sample_poisson (rng_t & rng, float mean)
{
    std::poisson_distribution<int> sampler(mean);
    return sampler(rng);
}

inline int sample_negative_binomial (rng_t & rng, float p, int r)
{
    std::negative_binomial_distribution<int> sampler(r, p);
    return sampler(rng);
}

inline float sample_gamma (
        rng_t & rng,
        float alpha,
        float beta = 1.f)
{
    // HACK <float> appears to be broken in libstdc++ 4.6
    //typedef std::gamma_distribution<float> gamma_distribution_t;
    typedef std::gamma_distribution<double> gamma_distribution_t;

    gamma_distribution_t sampler(alpha, beta);
    return sampler(rng);
}

inline float sample_beta (
        rng_t & rng,
        float alpha,
        float beta)
{
    float x = sample_gamma(rng, alpha, 1);
    float y = sample_gamma(rng, beta, 1);
    return x / (x + y);
}

void sample_dirichlet (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs);

void sample_dirichlet_safe (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs,
        float min_value);

inline float fast_score_student_t (
        float x,
        float nu,
        float mu,
        float lambda)
{
    // \cite{murphy2007conjugate}, Eq. 304
    float p = 0.f;
    p += fast_lgamma_nu(nu);
    p += 0.5f * fast_log(lambda / (M_PIf * nu));
    p += (-0.5f * nu - 0.5f) * fast_log(1.f + (lambda * sqr(x - mu)) / nu);
    return p;
}

inline float score_student_t (
        float x,
        float nu,
        float mu,
        float lambda)
{
    // \cite{murphy2007conjugate}, Eq. 304
    float p = lgammaf(nu * 0.5f + 0.5f) - lgammaf(nu * 0.5f);
    p += 0.5f * logf(lambda / (M_PIf * nu));
    p += (-0.5f * nu - 0.5f) * logf(1.f + (lambda * sqr(x - mu)) / nu);
    return p;
}

template<class T>
inline T sample_from_urn (
        rng_t & rng,
        const std::vector<T> & urn)
{
    DIST_ASSERT(urn.size() >= 1, "urn is too small to sample from");
    size_t f = sample_int(rng, 0, urn.size() - 1);
    DIST_ASSERT(0 <= f and f < urn.size(), "bad value: " << f);
    return urn[f];
}

template<class T>
inline std::pair<T, T> sample_pair_from_urn (
        rng_t & rng,
        const std::vector<T> & urn)
{
    DIST_ASSERT(urn.size() >= 2, "urn is too small to sample pair from");
    size_t f1 = sample_int(rng, 0, urn.size() - 1);
    size_t f2 = sample_int(rng, 0, urn.size() - 2);
    if (f2 >= f1) {
        f2 += 1;
    }
    DIST_ASSERT(0 <= f1 and f1 < urn.size(), "bad value: " << f1);
    DIST_ASSERT(0 <= f2 and f2 < urn.size(), "bad value: " << f2);
    DIST_ASSERT(f1 != f2, "bad pair: " << f1 << ", " << f2);
    return std::make_pair(urn[f1], urn[f2]);
}


//----------------------------------------------------------------------------
// Discrete Distribution
//
// Terminology:
//
//         prob = probability
//   likelihood = non-normalized probability
//        score = non-normalized log probability

template<class Alloc>
float log_sum_exp (const std::vector<float, Alloc> & scores);

inline size_t sample_discrete (
        rng_t & rng,
        size_t dim,
        const float * probs)
{
    float t = sample_unif01(rng);
    for (size_t i = 0; DIST_LIKELY(i < dim - 1); ++i) {
        t -= probs[i];
        if (DIST_UNLIKELY(t < 0)) {
            return i;
        }
    }
    return dim - 1;
}

template<class Alloc>
inline size_t sample_from_likelihoods (
        rng_t & rng,
        const std::vector<float, Alloc> & likelihoods,
        float total_likelihood)
{
    const size_t size = likelihoods.size();

    float t = total_likelihood * sample_unif01(rng);

    for (size_t i = 0; DIST_LIKELY(i < size); ++i) {
        t -= likelihoods[i];
        if (DIST_UNLIKELY(t < 0)) {
            return i;
        }
    }

    return size - 1;
}

template<class Alloc>
inline size_t sample_from_likelihoods (
        rng_t & rng,
        const std::vector<float, Alloc> & likelihoods)
{
    float total = vector_sum(likelihoods.size(), likelihoods.data());
    return sample_from_likelihoods(rng, likelihoods, total);
}

template<class Alloc>
inline size_t sample_from_probs (
        rng_t & rng,
        const std::vector<float, Alloc> & probs)
{
    return sample_from_likelihoods(rng, probs, 1.f);
}

// returns total likelihood
template<class Alloc>
float scores_to_likelihoods (std::vector<float, Alloc> & scores);

template<class Alloc>
void scores_to_probs (std::vector<float, Alloc> & scores)
{
    float total = scores_to_likelihoods(scores);
    vector_scale(scores.size(), scores.data(), 1.f / total);
}

template<class Alloc>
inline size_t sample_from_scores_overwrite (
        rng_t & rng,
        std::vector<float, Alloc> & scores)
{
    float total = scores_to_likelihoods(scores);
    return sample_from_likelihoods(rng, scores, total);
}

template<class Alloc>
inline std::pair<size_t, float> sample_prob_from_scores_overwrite (
        rng_t & rng,
        std::vector<float, Alloc> & scores)
{
    float total = scores_to_likelihoods(scores);
    size_t sample = sample_from_likelihoods(rng, scores, total);
    float prob = scores[sample] / total;
    return std::make_pair(sample, prob);
}

// score_from_scores_overwrite(...) = log(prob_from_scores_overwrite(...)),
// this is less succeptible to overflow than prob_from_scores_overwrite
template<class Alloc>
float score_from_scores_overwrite (
        rng_t & rng,
        size_t sample,
        std::vector<float, Alloc> & scores);

template<class Alloc>
inline size_t sample_from_scores (
        rng_t & rng,
        const std::vector<float, Alloc> & scores)
{
    std::vector<float, Alloc> scores_copy(scores);
    return sample_from_scores_overwrite(rng, scores_copy);
}

} // namespace distributions
