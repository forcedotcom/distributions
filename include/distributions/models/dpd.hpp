#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/sparse_counter.hpp>

namespace distributions
{

struct DirichletProcessDiscrete
{

//----------------------------------------------------------------------------
// Data

typedef std::vector<float> betas_t;  // dense

float gamma;
float alpha;
float beta0;
betas_t betas;

//----------------------------------------------------------------------------
// Datatypes

typedef uint32_t Value;

struct Group
{
    typedef SparseCounter<Value, uint32_t> counts_t;  // sparse

    counts_t counts;
};

struct Sampler
{
    typedef std::vector<float> probs_t;  // dense

    probs_t probs;
};

struct Scorer
{
    typedef std::vector<float> scores_t;  // dense

    scores_t scores;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        Group & group,
        rng_t &) const
{
    group.counts.clear();
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
   group.counts.add(value);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
   group.counts.remove(value);
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    destin.counts.merge(source.counts);
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        Sampler & sampler,
        const Group & group,
        rng_t & rng) const
{
    std::vector<float> & probs = sampler.probs;
    probs.clear();
    probs.reserve(betas.size() + 1);
    for (float beta : betas) {
        probs.push_back(beta * alpha);
    }
    for (auto i : group.counts) {
        probs[i.first] += i.second;
    }
    probs.push_back(beta0 * alpha);

    sample_dirichlet(rng, probs.size(), probs.data(), probs.data());
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_discrete(rng, sampler.probs.size(), sampler.probs.data());
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
    const size_t size = betas.size();
    const size_t total = group.counts.get_total();
    auto & scores = scorer.scores;
    scores.resize(size);

    const float betas_scale = alpha / (alpha + total);
    for (size_t i = 0; i < size; ++i) {
        scores[i] = betas_scale * betas[i];
    }

    const float counts_scale = 1.0f / (alpha + total);
    for (auto i : group.counts) {
        Value value = i.first;
        DIST_ASSERT(value < size,
            "unknown DPM value: " << value << " >= " << size);
        scores[value] += counts_scale * i.second;
    }
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    const auto & scores = scorer.scores;
    size_t size = scores.size();
    DIST_ASSERT(value < size,
        "unknown DPM value: " << value << " >= " << size);
    return fast_log(scores[value]);
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
    const size_t size = betas.size();
    const size_t total = group.counts.get_total();

    float score = 0;
    for (auto i : group.counts) {
        Value value = i.first;
        DIST_ASSERT(value < size,
            "unknown DPM value: " << value << " >= " << size);
        float prior_i = betas[value] * alpha;
        score += fast_lgamma(prior_i + i.second)
               - fast_lgamma(prior_i);
    }
    score += fast_lgamma(alpha)
           - fast_lgamma(alpha + total);

    return score;
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
