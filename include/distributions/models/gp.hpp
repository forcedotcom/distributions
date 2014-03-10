#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>

namespace distributions
{

struct GammaPoisson
{

//----------------------------------------------------------------------------
// Data

float alpha;
float inv_beta;

//----------------------------------------------------------------------------
// Datatypes

typedef GammaPoisson Model;

typedef uint32_t Value;

struct Group
{
    uint32_t count;
    uint32_t sum;
    float log_prod;
};

struct Sampler
{
    float mean;
};

struct Scorer
{
    float score;
    float post_alpha;
    float score_coeff;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (Group & group, rng_t &) const
{
    group.count = 0;
    group.sum = 0;
    group.log_prod = 0.f;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.count;
    group.sum += value;
    group.log_prod += fast_log_factorial(value);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    --group.count;
    group.sum -= value;
    group.log_prod -= fast_log_factorial(value);
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    destin.count += source.count;
    destin.sum += source.sum;
    destin.log_prod += source.log_prod;
}

Model plus_group (const Group & group) const
{
    Model post;
    post.alpha = alpha + group.sum;
    post.inv_beta = inv_beta + group.count;
    return post;
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        Sampler & sampler,
        const Group & group,
        rng_t & rng) const
{
    Model post = plus_group(group);
    sampler.mean = sample_gamma(rng, post.alpha, 1.f / post.inv_beta);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_poisson(rng, sampler.mean);
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
    Model post = plus_group(group);
    float score_coeff = -fast_log(1.f + post.inv_beta);
    scorer.score = -fast_lgamma(post.alpha)
                 + post.alpha * (fast_log(post.inv_beta) + score_coeff);
    scorer.post_alpha = post.alpha;
    scorer.score_coeff = score_coeff;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return scorer.score
         + fast_lgamma(scorer.post_alpha + value)
         - fast_log_factorial(value)
         + scorer.score_coeff * value;
}

float score_group (
        const Group & group,
        rng_t &) const
{
    Model post = plus_group(group);

    float score = fast_lgamma(post.alpha) - fast_lgamma(alpha);
    score += alpha * fast_log(inv_beta) - post.alpha * fast_log(post.inv_beta);
    score += -group.log_prod;
    return score;

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

}; // struct GammaPoisson

} // namespace distributions
