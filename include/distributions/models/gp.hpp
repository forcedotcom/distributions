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

struct GammaPoisson
{

//----------------------------------------------------------------------------
// Data

float alpha;
float inv_beta;

//----------------------------------------------------------------------------
// Datatypes

typedef GammaPoisson Model;

typedef uint32_t Value;

struct Group
{
    uint32_t count;
    uint32_t sum;
    float log_prod;
};

struct Sampler
{
    float mean;
};

struct Scorer
{
    float score;
    float post_alpha;
    float score_coeff;
};

struct Classifier
{
    std::vector<Group> groups;
    VectorFloat score;
    VectorFloat post_alpha;
    VectorFloat score_coeff;
    mutable VectorFloat temp;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (Group & group, rng_t &) const
{
    group.count = 0;
    group.sum = 0;
    group.log_prod = 0.f;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    ++group.count;
    group.sum += value;
    group.log_prod += fast_log_factorial(value);
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    --group.count;
    group.sum -= value;
    group.log_prod -= fast_log_factorial(value);
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    destin.count += source.count;
    destin.sum += source.sum;
    destin.log_prod += source.log_prod;
}

Model plus_group (const Group & group) const
{
    Model post;
    post.alpha = alpha + group.sum;
    post.inv_beta = inv_beta + group.count;
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
    sampler.mean = sample_gamma(rng, post.alpha, 1.f / post.inv_beta);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_poisson(rng, sampler.mean);
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
    float score_coeff = -fast_log(1.f + post.inv_beta);
    scorer.score = -fast_lgamma(post.alpha)
                 + post.alpha * (fast_log(post.inv_beta) + score_coeff);
    scorer.post_alpha = post.alpha;
    scorer.score_coeff = score_coeff;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    return scorer.score
         + fast_lgamma(scorer.post_alpha + value)
         - fast_log_factorial(value)
         + scorer.score_coeff * value;
}

float score_group (
        const Group & group,
        rng_t &) const
{
    Model post = plus_group(group);

    float score = fast_lgamma(post.alpha) - fast_lgamma(alpha);
    score += alpha * fast_log(inv_beta) - post.alpha * fast_log(post.inv_beta);
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
    classifier.post_alpha[groupid] = scorer.post_alpha;
    classifier.score_coeff[groupid] = scorer.score_coeff;
}

void _classifier_resize (
        Classifier & classifier,
        size_t group_count) const
{
    classifier.groups.resize(group_count);
    classifier.score.resize(group_count);
    classifier.post_alpha.resize(group_count);
    classifier.score_coeff.resize(group_count);
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
        classifier.post_alpha[groupid] = classifier.post_alpha.back();
        classifier.score_coeff[groupid] = classifier.score_coeff.back();
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

}; // struct GammaPoisson

} // namespace distributions
