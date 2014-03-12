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

int dim;  // fixed parameter
float alphas[max_dim];  // hyperparamter

//----------------------------------------------------------------------------
// Datatypes

typedef int Value;

struct Group
{
    uint32_t counts[max_dim];
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

struct VectorScorer
{
    std::vector<VectorFloat> scores;
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
        sampler.ps[i] = alphas[i] + group.counts[i];
    }

    sample_dirichlet(rng, dim, sampler.ps, sampler.ps);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_discrete(rng, dim, sampler.ps);
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
        float alpha = alphas[i] + group.counts[i];
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
    uint32_t count_sum = 0;
    float alpha_sum = 0;
    float score = 0;

    for (int i = 0; i < dim; ++i) {
        uint32_t count = group.counts[i];
        float alpha = alphas[i];
        count_sum += count;
        alpha_sum += alpha;
        score += fast_lgamma(alpha + count) - fast_lgamma(alpha);
    }

    score += fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + count_sum);

    return score;
}

void vector_scorer_init (
        VectorScorer & scorer,
        size_t group_count,
        rng_t & rng) const
{
    scorer.scores.resize(dim);
    for (int i = 0; i < dim; ++i) {
        scorer.scores[i].resize(group_count);
    }
}

void vector_scorer_update (
        VectorScorer & scorer,
        size_t group_index,
        const Group & group,
        rng_t &) const
{
    VectorFloat & scores = scorer.scores[group_index];
    float alpha_sum = 0;
    for (int i = 0; i < dim; ++i) {
        float alpha = alphas[i] + group.counts[i];
        scores[i] = alpha;
        alpha_sum += alpha;
    }
    float shift = -fast_log(alpha_sum);
    vector_log(scores.size(), scores.data());
    vector_shift(scores.size(), scores.data(), shift);
}


void vector_scorer_eval (
        VectorFloat & scores,
        const VectorScorer & scorer,
        const Value & value,
        rng_t &) const
{
    scores = scorer.scores[value];
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
