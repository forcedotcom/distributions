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
    uint32_t dpcount;   // = data point count
    float sampmean;     // = sample mean
    float allsampvar;   // = dpcount * sample variance
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

void group_init(
        Group & group,
        rng_t &) const
{
    group.dpcount = 0;
    group.sampmean = 0.f;
    group.allsampvar = 0.f;
}

void group_add_value(
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.dpcount;
    float delta = value - group.sampmean;
    group.sampmean += delta / group.dpcount;
    group.allsampvar += delta * (value - group.sampmean);
}

void group_remove_value(
        Group & group,
        const Value & value,
        rng_t &) const
{
    float total = group.sampmean * group.dpcount;
    float delta = value - group.sampmean;
    DIST_ASSERT(group.dpcount == 0, "Can't remove empty group");

    --group.dpcount;
    if (group.dpcount == 0) {
        group.sampmean = 0.f;
    } else {
        group.sampmean = (total - value) / group.dpcount;
    }
    if(group.dpcount <= 1) {
        group.allsampvar = 0.f;
    } else {
        group.allsampvar -= delta * (value - group.sampmean);
    }
}

void group_merge(
        Group & destin,
        const Group & source,
        rng_t &) const
{
    uint32_t dpcount = destin.dpcount + source.dpcount;
    float delta = source.sampmean - destin.sampmean;
    float source_part = float(source.dpcount) / dpcount;
    float cross_part = destin.dpcount * source_part;
    destin.dpcount = dpcount;
    destin.sampmean += source_part * delta;
    destin.allsampvar += source.allsampvar + cross_part * sqr(delta);
}

//----------------------------------------------------------------------------
// Sampling

//----------------------------------------------------------------------------
// Scoring

void scorer_init(
        Scorer & scorer,
        const Group & group,
        rng_t &) const
{
    float n = group.dpcount;
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

float scorer_eval(
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return scorer.score + scorer.log_coeff * fast_log(
         1.f + scorer.precision * sqr(value - scorer.mean));
}

float score_value(
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    Scorer scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group(
        const Group & group,
        rng_t &) const
{
    float n = group.dpcount;
    float sampmean = group.sampmean;
    float allsampvar = group.allsampvar;

    float kappa_n = kappa + n;

    float nu_n = nu + n;
    float sigmasq_n = 1.f / (nu_n) * ( sigmasq * nu + allsampvar +
                                       kappa * n / (kappa + n) *
                                       (sampmean - mu) * (sampmean - mu));
    float log_pi = 1.1447298858493991f;

    float score = 0.f;
    score += fast_lgamma(nu_n / 2.f) - fast_lgamma(nu / 2.f);
    score += 0.5f * fast_log(kappa / kappa_n);
    score += nu / 2.f * (fast_log(nu * sigmasq))
           - nu_n / 2.f * fast_log(nu_n * sigmasq_n);
    score += -n / 2.f * log_pi;
    return score;
}

}; // struct NormalInverseChiSq

} // namespace distributions
