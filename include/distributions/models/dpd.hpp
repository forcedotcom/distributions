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
#include <distributions/sparse_counter.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>

namespace distributions
{

struct DirichletProcessDiscrete
{

//----------------------------------------------------------------------------
// Data

float gamma;
float alpha;
float beta0;
std::vector<float> betas;  // dense

//----------------------------------------------------------------------------
// Datatypes

typedef uint32_t count_t;

typedef uint32_t Value;

static constexpr Value OTHER () { return -1; }

struct Group
{
    SparseCounter<Value, count_t> counts;  // sparse
};

struct Sampler
{
    std::vector<float> probs;  // dense
};

struct Scorer
{
    std::vector<float> scores;  // dense
};

struct Classifier
{
    std::vector<Group> groups;
    std::vector<VectorFloat> scores;  // dense
    VectorFloat scores_shift;
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

//----------------------------------------------------------------------------
// Classification

void classifier_init (
        Classifier & classifier,
        rng_t &) const
{
    const Value dim = betas.size();
    const size_t group_count = classifier.groups.size();
    classifier.scores_shift.resize(group_count);
    classifier.scores.resize(dim);
    for (Value value = 0; value < dim; ++value) {
        classifier.scores[value].resize(group_count);
    }
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        const Group & group = classifier.groups[groupid];
        for (Value value = 0; value < dim; ++value) {
            classifier.scores[value][groupid] =
                alpha * betas[value] + group.counts.get_count(value);
        }
        classifier.scores_shift[groupid] = alpha + group.counts.get_total();
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
    const Value dim = betas.size();
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
    const Value dim = betas.size();
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
    DIST_ASSERT1(value < betas.size(), "value out of bounds");
    Group & group = classifier.groups[groupid];
    count_t count = group.counts.add(value);
    count_t count_sum = group.counts.get_total();
    classifier.scores[value][groupid] = fast_log(alpha * betas[value] + count);
    classifier.scores_shift[groupid] = fast_log(alpha + count_sum);
}

void classifier_remove_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(groupid < classifier.groups.size(), "groupid out of bounds");
    DIST_ASSERT1(value < betas.size(), "value out of bounds");
    Group & group = classifier.groups[groupid];
    count_t count = group.counts.remove(value);
    count_t count_sum = group.counts.get_total();
    classifier.scores[value][groupid] = fast_log(alpha * betas[value] + count);
    classifier.scores_shift[groupid] = fast_log(alpha + count_sum);
}

void classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const
{
    DIST_ASSERT1(value < betas.size(), "value out of bounds");
    const size_t group_count = classifier.groups.size();
    vector_add_subtract(
        group_count,
        scores_accum,
        classifier.scores[value].data(),
        classifier.scores_shift.data());
}

}; // struct DirichletDiscrete<max_dim>

} // namespace distributions
