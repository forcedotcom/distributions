#pragma once

#include <random>
#include <distributions/common.hpp>

namespace distributions
{

//typedef std::default_random_engine rng_t;
//typedef std::mt19937 rng_t;
typedef std::ranlux48 rng_t;


// HACK std::gamma_distribution<float> appears to be broken
//typedef std::gamma_distribution<float> gamma_distribution_t;
typedef std::gamma_distribution<double> gamma_distribution_t;

inline float sample_unif01 (rng_t & rng)
{
    std::uniform_real_distribution<float> sampler(0.0, 1.0);
    return sampler(rng);
}

inline float sample_gamma (
        float alpha,
        float beta,
        rng_t & rng)
{
    gamma_distribution_t sampler;
    gamma_distribution_t::param_type param(alpha, beta);
    return sampler(rng, param);
}

void sample_dirichlet (
        size_t dim,
        const float * alphas,
        float * ps,
        rng_t & rng);

int sample_discrete (
        size_t dim,
        const float * ps,
        rng_t & rng);

} // namespace distributions
