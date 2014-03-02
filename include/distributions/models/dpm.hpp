#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/sparse_counter.hpp>

namespace distributions
{

struct DirichletProcessMixture
{

//----------------------------------------------------------------------------
// Data

typedef std::vector<float> betas_t;  // dense

struct hypers_t
{
    float gamma;
    float alpha;
    float beta0;
    betas_t betas;
};

hypers_t hypers;

//----------------------------------------------------------------------------
// Datatypes

typedef uint32_t value_t;

struct group_t
{
    typedef SparseCounter<value_t, uint32_t> counts_t;  // sparse

    counts_t counts;
};

struct sampler_t
{
    typedef std::vector<float> probs_t;  // dense

    probs_t probs;
};

struct scorer_t
{
    typedef std::vector<float> scores_t;  // dense

    scores_t scores;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        group_t & group,
        rng_t &) const
{
    group.counts.clear();
}

void group_add_value (
        group_t & group,
        const value_t & value,
        rng_t &) const
{
   group.counts.add(value);
}

void group_remove_value (
        group_t & group,
        const value_t & value,
        rng_t &) const
{
   group.counts.remove(value);
}

void group_merge (
        group_t & destin,
        const group_t & source,
        rng_t &) const
{
    destin.counts.merge(source.counts);
}

//----------------------------------------------------------------------------
// Sampling

#if 0
void sampler_init (
        sampler_t & sampler,
        const group_t & group,
        rng_t & rng) const
{
    for (int i = 0; i < dim; ++i) {
        sampler.ps[i] = alphas[i] + group.counts[i];
    }

    sample_dirichlet(dim, sampler.ps, sampler.ps, rng);
}

value_t sampler_eval (
        const sampler_t & sampler,
        rng_t & rng) const
{
    return sample_discrete(dim, sampler.ps, rng);
}

value_t sample_value (
        const group_t & group,
        rng_t & rng) const
{
    sampler_t sampler;
    sampler_init(sampler, group, rng);
    return sampler_eval(sampler, rng);
}
#endif

//----------------------------------------------------------------------------
// Scoring

void scorer_init (
        scorer_t & scorer,
        const group_t & group,
        rng_t &) const
{
    const size_t size = hypers.betas.size();
    const size_t total = group.counts.get_total();
    auto & scores = scorer.scores;
    scores.resize(size);

    const float betas_scale = hypers.alpha / (hypers.alpha + total);
    for (size_t i = 0; i < size; ++i) {
        scores[i] = betas_scale * hypers.betas[i];
    }

    const float counts_scale = 1.0f / (hypers.alpha + total);
    for (auto i : group.counts) {
        value_t value = i.first;
        DIST_ASSERT(value < size,
            "unknown DPM value: " << value << " >= " << size);
        scores[value] += counts_scale * i.second;
    }
}

float scorer_eval (
        const scorer_t & scorer,
        const value_t & value,
        rng_t &) const
{
    const auto & scores = scorer.scores;
    size_t size = scores.size();
    DIST_ASSERT(value < size,
        "unknown DPM value: " << value << " >= " << size);
    return fastlog(scores[value]);
}

float score_value (
        const group_t & group,
        const value_t & value,
        rng_t & rng) const
{
    scorer_t scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group (
        const group_t & group,
        rng_t &) const
{
    const size_t size = hypers.betas.size();
    const size_t total = group.counts.get_total();

    float score = 0;
    for (auto i : group.counts) {
        value_t value = i.first;
        DIST_ASSERT(value < size,
            "unknown DPM value: " << value << " >= " << size);
        float prior_i = hypers.betas[value] * hypers.alpha;
        score += fastlgamma(prior_i + i.second)
               - fastlgamma(prior_i);
    }
    score += fastlgamma(hypers.alpha)
           - fastlgamma(hypers.alpha + total);

    return score;
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
