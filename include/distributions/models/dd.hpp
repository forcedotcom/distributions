#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>

namespace distributions
{

template<int max_dim>
struct DirichletDiscrete
{

//                          Comments are only temporary
//----------------------------------------------------------------------------
// Data

int dim;                    // fixed paramter
float alphas[max_dim];      // prior hyperparameter

//----------------------------------------------------------------------------
// Datatypes

typedef int value_t;        // per-row state

struct group_t              // local per-component state
{
    int counts[max_dim];    // sufficient statistic
};

struct sampler_t            // partially evaluated sample_value function
{
    float ps[max_dim];
};

struct scorer_t             // partially evaluated score_value function
{
    float alpha_sum;
    float alphas[max_dim];
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        group_t & group,
        rng_t &)
{
    for (int i = 0; i < dim; ++i) {
        group.counts[i] = 0;
    }
}

void group_add_data (
        group_t & group,
        const value_t & value,
        rng_t &)
{
   group.counts[value] += 1;
}

void group_remove_data (
        group_t & group,
        const value_t & value,
        rng_t &) const
{
   group.counts[value] -= 1;
}

void group_merge (
        group_t & destin,
        const group_t & source,
        rng_t &) const
{
    for (int i = 0; i < dim; ++i) {
        destin.counts[i] += source.counts[i];
    }
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        sampler_t & sampler,
        const group_t & group,
        rng_t & rng)
{
    for (int i = 0; i < dim; ++i) {
        sampler.ps[i] = alphas[i] + group.counts[i];
    }

    sample_dirichlet(dim, sampler.ps, sampler.ps, rng);
}

value_t sampler_eval (
        const sampler_t & sampler,
        rng_t & rng)
{
    return sample_discrete(dim, sampler.ps, rng);
}

value_t sample_value (
        const group_t & group,
        rng_t & rng)
{
    sampler_t sampler;
    sampler_init(sampler, group, rng);
    return sampler_eval(sampler, rng);
}

//----------------------------------------------------------------------------
// Scoring

void scorer_init (
        scorer_t & scorer,
        const group_t & group,
        rng_t &)
{
    float alpha_sum = 0;

    for (int i = 0; i < dim; ++i) {
        float alpha = alphas[i] + group.counts[i];
        scorer.alphas[i] = alpha;
        alpha_sum += alpha;
    }

    scorer.alpha_sum = alpha_sum;
}

float scorer_eval (
        const scorer_t & scorer,
        const value_t & value,
        rng_t &)
{
    return fastlog(scorer.alphas[value] / scorer.alpha_sum);
}

float score_value (
        const group_t & group,
        const value_t & value,
        rng_t & rng)
{
    scorer_t scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group (
        const group_t & group,
        rng_t &)
{
    int count_sum = 0;
    float alpha_sum = 0;
    float score = 0;

    for (int i = 0; i < dim; ++i) {
        int count = group.counts[i];
        float alpha = alphas[i];
        count_sum += count;
        alpha_sum += alpha;
        score += fastlgamma(alpha + count) - fastlgamma(alpha);
    }

    score += fastlgamma(alpha_sum) - fastlgamma(alpha_sum + count_sum);

    return score;
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
