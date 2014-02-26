#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>

namespace distributions
{

template<int max_dim>
struct DirichletDiscrete
{

//----------------------------------------------------------------------------
// Datatypes                Comments are only temporary

typedef int value_t;        // per-row state

struct model_t              // global state
{
    int dim;                // paramter
    float alphas[max_dim];  // hyperparameter
};

struct group_t              // local per-component state
{
    int counts[max_dim];    // sufficient statistic
};

struct sampler_t            // partially evaluated sample_value function
{
    int dim;
    float ps[max_dim];
};

struct scorer_t             // partially evaluated score_value function
{
    float alpha_sum;
    float alphas[max_dim];
};

//----------------------------------------------------------------------------
// Mutation

static void group_init_prior (
        group_t & group,
        const model_t & model,
        rng_t &)
{
    for (int i = 0, dim = model.dim; i < dim; ++i) {
        group.counts[i] = 0;
    }
}

static void group_add_data (
        const value_t & value,
        group_t & group,
        const model_t & model)
{
   group.counts[value] += 1;
}

static void group_remove_data (
        const value_t & value,
        group_t & group,
        const model_t & model)
{
   group.counts[value] -= 1;
}

static void group_merge (
        group_t & destin,
        const group_t & source,
        const model_t & model)
{
    destin.count_sum += source.count_sum;
    for (int i = 0, dim = model.dim; i < dim; ++i) {
        destin.counts[i] += source.counts[i];
    }
}

//----------------------------------------------------------------------------
// Sampling

static void sampler_init (
        sampler_t & sampler,
        const group_t & group,
        const model_t & model,
        rng_t & rng)
{
    sampler.dim = model.dim;

    for (int i = 0, dim = model.dim; i < dim; ++i) {
        sampler.ps[i] = model.alphas[i] + group.counts[i];
    }

    sample_dirichlet(model.dim, sampler.ps, sampler.ps, rng);
}

static value_t sampler_eval (
        const sampler_t & sampler,
        rng_t & rng)
{
    return sample_multinomial(sampler.dim, sampler.ps);
}

static value_t sample_value (
        const group_t & group,
        const model_t & model,
        rng_t & rng)
{
    sampler_t sampler;
    sampler_init(sampler, group, model);
    return sampler_eval(sampler, rng);
}

//----------------------------------------------------------------------------
// Scoring

static void scorer_init (
        scorer_t & scorer,
        const group_t & group,
        const model_t & model,
        rng_t &)
{
    float alpha_sum = 0;

    for (int i = 0, dim = model.dim; i < dim; ++i) {
        float alpha = model.alphas[i] + group.counts[i];
        scorer.alphas[i] = alpha;
        alpha_sum += alpha;
    }

    scorer.alpha_sum = alpha_sum;
}

static float scorer_eval (
        const value_t & value,
        const scorer_t & scorer,
        rng_t &)
{
    return fastlog(scorer->alphas[value] / scorer->alpha_sum);
}

static float score_value (
        const value_t & value,
        const group_t & group,
        const model_t & model,
        rng_t & rng)
{
    scorer_t scorer;
    scorer_init(scorer, group, model);
    return scorer_eval(value, scorer, rng);
}

static float score_group (
        const group_t & group,
        const model_t & model,
        rng_t &)
{
    int count_sum = 0;
    float alpha_sum = 0;
    float score = 0;

    for (int i = 0, dim = model.dim; i < dim; ++i) {
        int count = group.counts[i];
        float alpha = model.alphas[i];
        count_sum += count;
        alpha_sum += alpha;
        score += lgamma(alpha + count) - lgamma(alpha);
    }

    score += lgamma(alpha_sum) - lgamma(alpha_sum + count_sum);

    return score;
}

}; // struct AsymmetricDirichletDiscrete<max_dim>

} // namespace distributions
