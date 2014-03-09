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
float beta;

//----------------------------------------------------------------------------
// Datatypes

typedef uint32_t Value;

struct Group
{
    uint32_t sum;
    float log_prod;
    uint32_t n;
};

struct Scorer
{
    float score;
    float alpha_n;
    float score_coeff;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (Group & group, rng_t &) const
{
    group.sum = 0;
    group.log_prod = 0.f;
    group.n = 0;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.n;
    group.sum += value;
    group.log_prod += fast_log_factorial(value);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.n;
    group.sum -= value;
    group.log_prod -= fast_log_factorial(value);
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    destin.n += source.n;
    destin.sum += source.sum;
    destin.log_prod += source.log_prod;
}

//----------------------------------------------------------------------------
// Sampling

//----------------------------------------------------------------------------
// Scoring

void scorer_init (
        Scorer & scorer,
        const Group & group,
        rng_t &) const
{
    float alpha_n = alpha + group.sum;
    float inv_beta_n = group.n + 1.f / beta;
    float score_coeff = -fast_log(1.f + inv_beta_n);

    scorer.score = -fast_lgamma(alpha_n)
                 + alpha_n * (fast_log(inv_beta_n) + score_coeff);
    scorer.alpha_n = alpha_n;
    scorer.score_coeff = score_coeff;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return scorer.score
         + fast_lgamma(scorer.alpha_n + value)
         - fast_log_factorial(value)
         + scorer.score_coeff * value;
}

float score_group (
        const Group & group,
        rng_t &) const
{
    float alpha_n = alpha + group.sum;
    float beta_n = 1.f / (group.n + 1.f / beta);

    float score = fast_lgamma(alpha_n) - fast_lgamma(alpha);
    score += alpha_n * fast_log(beta_n) - alpha * fast_log(beta);
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
