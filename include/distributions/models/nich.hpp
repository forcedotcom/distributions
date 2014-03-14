// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>

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

struct Classifier
{
    std::vector<Group> groups;
    VectorFloat score;
    VectorFloat log_coeff;
    VectorFloat precision;
    VectorFloat mean;
    mutable VectorFloat temp;
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

//----------------------------------------------------------------------------
// Classification

private:

void _classifier_update_group (
        Classifier & classifier,
        size_t groupid,
        rng_t & rng) const
{
    const Group & group = classifier.groups[groupid];
    Scorer scorer;
    scorer_init(scorer, group, rng);
    classifier.score[groupid] = scorer.score;
    classifier.log_coeff[groupid] = scorer.log_coeff;
    classifier.precision[groupid] = scorer.precision;
    classifier.mean[groupid] = scorer.mean;
}

void _classifier_resize (
        Classifier & classifier,
        size_t group_count) const
{
    classifier.groups.resize(group_count);
    classifier.score.resize(group_count);
    classifier.log_coeff.resize(group_count);
    classifier.precision.resize(group_count);
    classifier.mean.resize(group_count);
    classifier.temp.resize(group_count);
}

public:

void classifier_init (
        Classifier & classifier,
        rng_t & rng) const
{
    const size_t group_count = classifier.groups.size();
    _classifier_resize(classifier, group_count);
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        _classifier_update_group(classifier, groupid, rng);
    }
}

void classifier_add_group (
        Classifier & classifier,
        rng_t & rng) const
{
    const size_t groupid = classifier.groups.size();
    const size_t group_count = groupid + 1;
    _classifier_resize(classifier, group_count);
    group_init(classifier.groups.back(), rng);
    _classifier_update_group(classifier, groupid, rng);
}

void classifier_remove_group (
        Classifier & classifier,
        size_t groupid) const
{
    const size_t group_count = classifier.groups.size() - 1;
    if (groupid != group_count) {
        std::swap(classifier.groups[groupid], classifier.groups.back());
        classifier.score[groupid] = classifier.score.back();
        classifier.log_coeff[groupid] = classifier.log_coeff.back();
        classifier.precision[groupid] = classifier.precision.back();
        classifier.mean[groupid] = classifier.mean.back();
    }
    _classifier_resize(classifier, group_count);
}

void classifier_add_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
    DIST_ASSERT1(groupid < classifier.groups.size(), "groupid out of bounds");
    Group & group = classifier.groups[groupid];
    group_add_value(group, value, rng);
    _classifier_update_group(classifier, groupid, rng);
}

void classifier_remove_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
    DIST_ASSERT1(groupid < classifier.groups.size(), "groupid out of bounds");
    Group & group = classifier.groups[groupid];
    group_remove_value(group, value, rng);
    _classifier_update_group(classifier, groupid, rng);
}

void classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const;

}; // struct NormalInverseChiSq

} // namespace distributions
