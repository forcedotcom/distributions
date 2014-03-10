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

typedef float Value;

struct Group
{
    uint32_t count;   // = data point count
    float sampmean;     // = sample mean
    float allsampvar;   // = count * sample variance
};

struct Sampler
{
    float mu;
    float sigma;
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
    group.sampmean = 0.f;
    group.allsampvar = 0.f;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.count;
    float delta = value - group.sampmean;
    group.sampmean += delta / group.count;
    group.allsampvar += delta * (value - group.sampmean);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    float total = group.sampmean * group.count;
    float delta = value - group.sampmean;
    DIST_ASSERT(group.count == 0, "Can't remove empty group");

    --group.count;
    if (group.count == 0) {
        group.sampmean = 0.f;
    } else {
        group.sampmean = (total - value) / group.count;
    }
    if (group.count <= 1) {
        group.allsampvar = 0.f;
    } else {
        group.allsampvar -= delta * (value - group.sampmean);
    }
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    uint32_t count = destin.count + source.count;
    float delta = source.sampmean - destin.sampmean;
    float source_part = float(source.count) / count;
    float cross_part = destin.count * source_part;
    destin.count = count;
    destin.sampmean += source_part * delta;
    destin.allsampvar += source.allsampvar + cross_part * sqr(delta);
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
    float n = group.count;
    float sampmean = group.sampmean;
    float allsampvar = group.allsampvar;

    float kappa_n = kappa + n;
    float mu_n = (kappa * mu + n * sampmean) / kappa_n;
    float nu_n = nu + n;
    float sigmasq_n = 1.f / nu_n * (sigmasq * nu + allsampvar +
                                    kappa * n / kappa_n *
                                    (sampmean - mu) * (sampmean - mu));

    float lambda = kappa_n / ((kappa_n + 1.f) * sigmasq_n);

    scorer.score = fast_lgamma_nu(nu_n)
                 + 0.5f * fast_log(lambda / (M_PIf * nu_n));
    scorer.log_coeff = -0.5f * nu_n - 0.5f;
    scorer.precision = lambda / nu_n;
    scorer.mean = mu_n;
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
    float n = group.count;
    float sampmean = group.sampmean;
    float allsampvar = group.allsampvar;

    float kappa_n = kappa + n;

    float nu_n = nu + n;
    float sigmasq_n = 1.f / nu_n * (
        sigmasq * nu +
        allsampvar +
        kappa * n / (kappa + n) * sqr(sampmean - mu));

    float log_pi = 1.1447298858493991f;

    float score = fast_lgamma(0.5f * nu_n) - fast_lgamma(0.5f * nu);
    score += 0.5f * fast_log(kappa / kappa_n);
    score += 0.5f * nu * (fast_log(nu * sigmasq))
           - 0.5f * nu_n * fast_log(nu_n * sigmasq_n);
    score += -0.5f * n * log_pi;
    return score;
}

}; // struct NormalInverseChiSq

} // namespace distributions
