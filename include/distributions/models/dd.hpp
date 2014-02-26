#pragma once

#include "../common.hpp"

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

struct score_add_fun_t      // partially evaluated score_add function
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

static void group_rem_data (
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
// Scoring

static void score_add_fun_init (
        score_add_fun_t & fun,
        const group_t & group,
        const model_t & model)
{
    float alpha_sum = 0;

    for (int i = 0, dim = model.dim; i < dim; ++i) {
        float alpha = model.alphas[i] + group.counts[i];
        fun.alphas[i] = alpha;
        alpha_sum += alpha;
    }

    fun.alpha_sum = alpha_sum;
}

static float score_add_eval (
        const value_t & value,
        const score_add_fun_t & fun,
        rng_t &)
{
    return fastlog(fun->alphas[value] / fun->alpha_sum);
}

static float score_add (
        const value_t & value,
        const group_t & group,
        const model_t & model,
        rng_t & rng) 
{
    score_add_fun_t fun;
    score_add_fun_init(fun, group, model);
    return score_add_eval(value, fun, rng);
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
