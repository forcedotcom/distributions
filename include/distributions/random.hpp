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

// HACK std::gamma_distribution<float> appears to be broken
//typedef std::gamma_distribution<float> gamma_distribution_t;
typedef std::gamma_distribution<double> gamma_distribution_t;

inline float sample_gamma (
        rng_t & rng,
        float alpha,
        float beta = 1.f)
{
    gamma_distribution_t sampler;
    gamma_distribution_t::param_type param(alpha, beta);
    return sampler(rng, param);
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

size_t sample_discrete (
        rng_t & rng,
        size_t dim,
        const float * probs);

size_t sample_from_likelihoods (
        rng_t & rng,
        const std::vector<float> & likelihoods,
        float total_likelihood);

inline size_t sample_from_likelihoods (
        rng_t & rng,
        const std::vector<float> & likelihoods)
{
    float total = vector_sum(likelihoods.size(), likelihoods.data());
    return sample_from_likelihoods(rng, likelihoods, total);
}

inline size_t sample_from_probs (
        rng_t & rng,
        const std::vector<float> & probs)
{
    return sample_from_likelihoods(rng, probs, 1.f);
}

// returns total likelihood
float scores_to_likelihoods (std::vector<float> & scores);

inline size_t sample_from_scores_overwrite (
        rng_t & rng,
        std::vector<float> & scores)
{
    float total = scores_to_likelihoods(scores);
    return sample_from_likelihoods(rng, scores, total);
}

inline std::pair<size_t, float> sample_prob_from_scores_overwrite (
        rng_t & rng,
        std::vector<float> & scores)
{
    float total = scores_to_likelihoods(scores);
    size_t sample = sample_from_likelihoods(rng, scores, total);
    float prob = scores[sample] / total;
    return std::make_pair(sample, prob);
}

// score_from_scores_overwrite(...) = log(prob_from_scores_overwrite(...)),
// this is less succeptible to overflow than prob_from_scores_overwrite
float score_from_scores_overwrite (
        rng_t & rng,
        size_t sample,
        std::vector<float> & scores);

inline size_t sample_from_scores (
        rng_t & rng,
        const std::vector<float> & scores)
{
    std::vector<float> scores_copy(scores);
    return sample_from_scores_overwrite(rng, scores_copy);
}

} // namespace distributions
