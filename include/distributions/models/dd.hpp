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

typedef uint32_t count_t;

typedef int Value;

struct Group
{
    count_t count_sum;
    count_t counts[max_dim];
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

struct Classifier
{
    std::vector<Group> groups;
    float alpha_sum;
    std::vector<VectorFloat> scores;
    VectorFloat scores_shift;
};

class Fitter
{
    std::vector<Group> groups;
    VectorFloat scores;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        Group & group,
        rng_t &) const
{
    group.count_sum = 0;
    for (Value value = 0; value < dim; ++value) {
        group.counts[value] = 0;
    }
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < dim, "value out of bounds");
    group.count_sum += 1;
    group.counts[value] += 1;
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < dim, "value out of bounds");
    group.count_sum -= 1;
    group.counts[value] -= 1;
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    DIST_ASSERT1(& destin != & source, "cannot merge with self");
    for (Value value = 0; value < dim; ++value) {
        destin.counts[value] += source.counts[value];
    }
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        Sampler & sampler,
        const Group & group,
        rng_t & rng) const
{
    for (Value value = 0; value < dim; ++value) {
        sampler.ps[value] = alphas[value] + group.counts[value];
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
    for (Value value = 0; value < dim; ++value) {
        float alpha = alphas[value] + group.counts[value];
        scorer.alphas[value] = alpha;
        alpha_sum += alpha;
    }
    scorer.alpha_sum = alpha_sum;
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < dim, "value out of bounds");
    return fast_log(scorer.alphas[value] / scorer.alpha_sum);
}

float score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    DIST_ASSERT1(value < dim, "value out of bounds");
    Scorer scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group (
        const Group & group,
        rng_t &) const
{
    float alpha_sum = 0;
    float score = 0;

    for (Value value = 0; value < dim; ++value) {
        float alpha = alphas[value];
        alpha_sum += alpha;
        score += fast_lgamma(alpha + group.counts[value]) - fast_lgamma(alpha);
    }

    score += fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + group.count_sum);

    return score;
}

//----------------------------------------------------------------------------
// Classification

void classifier_init (
        Classifier & classifier,
        rng_t &) const
{
    const size_t group_count = classifier.groups.size();
    classifier.scores_shift.resize(group_count);
    classifier.alpha_sum = 0;
    classifier.scores.resize(dim);
    for (Value value = 0; value < dim; ++value) {
        classifier.alpha_sum += alphas[value];
        classifier.scores[value].resize(group_count);
    }
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        const Group & group = classifier.groups[groupid];
        for (Value value = 0; value < dim; ++value) {
            classifier.scores[value][groupid] =
                alphas[value] + group.counts[value];
        }
        classifier.scores_shift[groupid] =
            classifier.alpha_sum + group.count_sum;
    }
    vector_log(group_count, classifier.scores_shift.data());
    for (Value value = 0; value < dim; ++value) {
        vector_log(group_count, classifier.scores[value].data());
    }
}

void classifier_add_group (
        Classifier & classifier,
        rng_t & rng) const
{
    const size_t group_count = classifier.groups.size() + 1;
    classifier.groups.resize(group_count);
    group_init(classifier.groups.back(), rng);
    classifier.scores_shift.resize(group_count, 0);
    for (Value value = 0; value < dim; ++value) {
        classifier.scores[value].resize(group_count, 0);
    }
}

void classifier_remove_group (
        Classifier & classifier,
        size_t groupid) const
{
    const size_t group_count = classifier.groups.size() - 1;
    if (groupid != group_count) {
        std::swap(classifier.groups[groupid], classifier.groups.back());
        classifier.scores_shift[groupid] = classifier.scores_shift.back();
        for (Value value = 0; value < dim; ++value) {
            VectorFloat & scores = classifier.scores[value];
            scores[groupid] = scores.back();
        }
    }
    classifier.groups.resize(group_count);
    classifier.scores_shift.resize(group_count);
    for (Value value = 0; value < dim; ++value) {
        classifier.scores[value].resize(group_count);
    }
}

void classifier_add_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(groupid < classifier.groups.size(), "groupid out of bounds");
    DIST_ASSERT1(value < dim, "value out of bounds");
    Group & group = classifier.groups[groupid];
    count_t count_sum = group.count_sum += 1;
    count_t count = group.counts[value] += 1;
    classifier.scores[value][groupid] = fast_log(alphas[value] + count);
    classifier.scores_shift[groupid] =
        fast_log(classifier.alpha_sum + count_sum);
}

void classifier_remove_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(groupid < classifier.groups.size(), "groupid out of bounds");
    DIST_ASSERT1(value < dim, "value out of bounds");
    Group & group = classifier.groups[groupid];
    count_t count_sum = group.count_sum -= 1;
    count_t count = group.counts[value] -= 1;
    classifier.scores[value][groupid] = fast_log(alphas[value] + count);
    classifier.scores_shift[groupid] =
        fast_log(classifier.alpha_sum + count_sum);
}

void classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const
{
    DIST_ASSERT1(value < dim, "value out of bounds");
    const size_t group_count = classifier.groups.size();
    vector_add_subtract(
        group_count,
        scores_accum,
        classifier.scores[value].data(),
        classifier.scores_shift.data());
}

//----------------------------------------------------------------------------
// Fitting

void fitter_init (
    Fitter & fitter) const
{
    const size_t group_count = fitter.groups.size();
    const float alpha_sum = vector_sum(dim, alphas);
    fitter.scores.resize(dim + 1);
    for (Value value = 0; value < dim; ++value) {
        vector_zero(fitter.scores[value].size(), fitter.scores[value].data());
    }
    float score_shift = 0;
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        const Group & group = fitter.groups[groupid];
        for (Value value = 0; value < dim; ++value) {
            float alpha = alphas[value];
            fitter.scores[value] +=
                fast_lgamma(alpha + group.counts[value]) - fast_lgamma(alpha);
        }
        score_shift +=
            fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + group.count_sum);
    }
    fitter.scores.back() = score_shift;
}

void fitter_set_param_alpha (
    Fitter & fitter,
    Value value,
    float alpha)
{
    DIST_ASSERT1(value < dim, "value out of bounds");

    alphas[value] = alpha;

    const size_t group_count = fitter.groups.size();
    const float alpha_sum = vector_sum(dim, alphas);
    float score = 0;
    float score_shift = 0;
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        const Group & group = fitter.groups[groupid];
        score += fast_lgamma(alpha + group.counts[value]) - fast_lgamma(alpha);
        score_shift +=
            fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + group.count_sum);
    }
    fitter.scores[value] = score;
    fitter.scores.back() = score_shift;
}

float fitter_score (
        Fitter & fitter) const
{
    return vector_sum(fitter.scores.size(), fitter.scores.data());
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
