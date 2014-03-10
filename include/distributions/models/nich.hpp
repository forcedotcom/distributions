#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>

namespace distributions
{

struct NormalInverseChiSq
{

//----------------------------------------------------------------------------
// Data

float mu;
float kappa;
float sigmasq;
float nu;

//----------------------------------------------------------------------------
// Datatypes

typedef NormalInverseChiSq Model;

typedef float Value;

struct Group
{
    uint32_t count;
    float mean;
    float count_times_variance;
};

struct Sampler
{
    float mu;
    float sigmasq;
};

struct Scorer
{
    float score;
    float log_coeff;
    float precision;
    float mean;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        Group & group,
        rng_t &) const
{
    group.count = 0;
    group.mean = 0.f;
    group.count_times_variance = 0.f;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.count;
    float delta = value - group.mean;
    group.mean += delta / group.count;
    group.count_times_variance += delta * (value - group.mean);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    float total = group.mean * group.count;
    float delta = value - group.mean;
    DIST_ASSERT(group.count > 0, "Can't remove empty group");

    --group.count;
    if (group.count == 0) {
        group.mean = 0.f;
    } else {
        group.mean = (total - value) / group.count;
    }
    if (group.count <= 1) {
        group.count_times_variance = 0.f;
    } else {
        group.count_times_variance -= delta * (value - group.mean);
    }
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    uint32_t count = destin.count + source.count;
    float delta = source.mean - destin.mean;
    float source_part = float(source.count) / count;
    float cross_part = destin.count * source_part;
    destin.count = count;
    destin.mean += source_part * delta;
    destin.count_times_variance +=
        source.count_times_variance + cross_part * sqr(delta);
}

Model plus_group (const Group & group) const
{
    Model post;
    float mu_1 = mu - group.mean;
    post.kappa = kappa + group.count;
    post.mu = (kappa * mu + group.mean * group.count) / post.kappa;
    post.nu = nu + group.count;
    post.sigmasq = 1.f / post.nu * (
        nu * sigmasq
        + group.count_times_variance
        + (group.count * kappa * mu_1 * mu_1) / post.kappa);
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
    sampler.sigmasq = post.nu * post.sigmasq / sample_chisq(rng, post.nu);
    sampler.mu = sample_normal(rng, post.mu, sampler.sigmasq / post.kappa);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_normal(rng, sampler.mu, sampler.sigmasq);
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
    float lambda = post.kappa / ((post.kappa + 1.f) * post.sigmasq);
    scorer.score =
        fast_lgamma_nu(post.nu) + 0.5f * fast_log(lambda / (M_PIf * post.nu));
    scorer.log_coeff = -0.5f * post.nu - 0.5f;
    scorer.precision = lambda / post.nu;
    scorer.mean = post.mu;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return scorer.score
         + scorer.log_coeff * fast_log(
             1.f + scorer.precision * sqr(value - scorer.mean));
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
    Model post = plus_group(group);
    float log_pi = 1.1447298858493991f;
    float score = fast_lgamma(0.5f * post.nu) - fast_lgamma(0.5f * nu);
    score += 0.5f * fast_log(kappa / post.kappa);
    score += 0.5f * nu * (fast_log(nu * sigmasq))
           - 0.5f * post.nu * fast_log(post.nu * post.sigmasq);
    score += -0.5f * group.count * log_pi;
    return score;
}

}; // struct NormalInverseChiSq

} // namespace distributions
