#pragma once

#include <utility>
#include <random>
#include <distributions/common.hpp>
#include <distributions/special.hpp>

namespace distributions
{

//typedef std::default_random_engine rng_t;
//typedef std::mt19937 rng_t;
typedef std::ranlux48 rng_t;


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

int sample_discrete (
        rng_t & rng,
        size_t dim,
        const float * probs);

inline float fast_score_student_t (
        float x,
        float v,
        float mean,
        float lambda)
{
    float p = 0.f;
    p += fast_lgamma_nu(v);
    p += 0.5f * fast_log(lambda / (M_PIf * v));
    p += (-0.5f * v - 0.5f) * fast_log(1.f + (lambda * sqr(x - mean)) / v);
    return p;
}

inline float score_student_t (
        float x,
        float v,
        float mean,
        float lambda)
{
    float p = 0.f;
    p += lgammaf(v * 0.5f + 0.5f) - lgammaf(v * 0.5f);
    p += 0.5f * logf(lambda / (M_PIf * v));
    p += (-0.5f * v - 0.5f) * logf(1.f + (lambda * sqr(x - mean)) / v);
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

} // namespace distributions
