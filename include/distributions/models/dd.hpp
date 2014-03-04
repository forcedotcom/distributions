#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>

namespace distributions
{

template<int max_dim>
struct DirichletDiscrete
{

//----------------------------------------------------------------------------
// Data

int dim;

struct Hypers
{
    float alphas[max_dim];
};

Hypers hypers;

//----------------------------------------------------------------------------
// Datatypes

typedef int Value;

struct Group
{
    int counts[max_dim];
};

struct Sampler
{
    float ps[max_dim];
};

struct Scorer
{
    float alpha_sum;
    float alphas[max_dim];
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        Group & group,
        rng_t &) const
{
    for (int i = 0; i < dim; ++i) {
        group.counts[i] = 0;
    }
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
   group.counts[value] += 1;
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
   group.counts[value] -= 1;
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    for (int i = 0; i < dim; ++i) {
        destin.counts[i] += source.counts[i];
    }
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        Sampler & sampler,
        const Group & group,
        rng_t & rng) const
{
    for (int i = 0; i < dim; ++i) {
        sampler.ps[i] = hypers.alphas[i] + group.counts[i];
    }

    sample_dirichlet(dim, sampler.ps, sampler.ps, rng);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_discrete(dim, sampler.ps, rng);
}

Value sample_value (
        const Group & group,
        rng_t & rng) const
{
    Sampler sampler;
    sampler_init(sampler, group, rng);
    return sampler_eval(sampler, rng);
}

//----------------------------------------------------------------------------
// Scoring

void scorer_init (
        Scorer & scorer,
        const Group & group,
        rng_t &) const
{
    float alpha_sum = 0;

    for (int i = 0; i < dim; ++i) {
        float alpha = hypers.alphas[i] + group.counts[i];
        scorer.alphas[i] = alpha;
        alpha_sum += alpha;
    }

    scorer.alpha_sum = alpha_sum;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return fast_log(scorer.alphas[value] / scorer.alpha_sum);
}

float score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    Scorer scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group (
        const Group & group,
        rng_t &) const
{
    int count_sum = 0;
    float alpha_sum = 0;
    float score = 0;

    for (int i = 0; i < dim; ++i) {
        int count = group.counts[i];
        float alpha = hypers.alphas[i];
        count_sum += count;
        alpha_sum += alpha;
        score += fast_lgamma(alpha + count) - fast_lgamma(alpha);
    }

    score += fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + count_sum);

    return score;
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
