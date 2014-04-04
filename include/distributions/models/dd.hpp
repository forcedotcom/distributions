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
typedef DirichletDiscrete<max_dim> Model;

static const char * name () { return "DirichletDiscrete"; }
static const char * short_name () { return "dd"; }

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

    void init (
            const Model & model,
            rng_t &)
    {
        count_sum = 0;
        for (Value value = 0; value < model.dim; ++value) {
            counts[value] = 0;
        }
    }

    void add_value (
            const Model & model,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        count_sum += 1;
        counts[value] += 1;
    }

    void remove_value (
            const Model & model,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        count_sum -= 1;
        counts[value] -= 1;
    }

    void merge (
            const Model & model,
            const Group & source,
            rng_t &)
    {
        for (Value value = 0; value < model.dim; ++value) {
            counts[value] += source.counts[value];
        }
    }
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

    void init (
            const Model & model,
            rng_t &)
    {
        const size_t group_count = groups.size();
        scores_shift.resize(group_count);
        alpha_sum = 0;
        scores.resize(model.dim);
        for (Value value = 0; value < model.dim; ++value) {
            alpha_sum += model.alphas[value];
            scores[value].resize(group_count);
        }
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = groups[groupid];
            for (Value value = 0; value < model.dim; ++value) {
                scores[value][groupid] =
                    model.alphas[value] + group.counts[value];
            }
            scores_shift[groupid] =
                alpha_sum + group.count_sum;
        }
        vector_log(group_count, scores_shift.data());
        for (Value value = 0; value < model.dim; ++value) {
            vector_log(group_count, scores[value].data());
        }
    }

    void add_group (
            const Model & model,
            rng_t & rng)
    {
        const size_t group_count = groups.size() + 1;
        groups.resize(group_count);
        groups.back().init(model, rng);
        scores_shift.resize(group_count, 0);
        for (Value value = 0; value < model.dim; ++value) {
            scores[value].resize(group_count, 0);
        }
    }

    void remove_group (
            const Model & model,
            size_t groupid)
    {
        const size_t group_count = groups.size() - 1;
        if (groupid != group_count) {
            std::swap(groups[groupid], groups.back());
            scores_shift[groupid] = scores_shift.back();
            for (Value value = 0; value < model.dim; ++value) {
                VectorFloat & vscores = scores[value];
                vscores[groupid] = vscores.back();
            }
        }
        groups.resize(group_count);
        scores_shift.resize(group_count);
        for (Value value = 0; value < model.dim; ++value) {
            scores[value].resize(group_count);
        }
    }

    void add_value (
            const Model & model,
            size_t groupid,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(groupid < groups.size(), "groupid out of bounds");
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        Group & group = groups[groupid];
        count_t count_sum = group.count_sum += 1;
        count_t count = group.counts[value] += 1;
        scores[value][groupid] = fast_log(model.alphas[value] + count);
        scores_shift[groupid] =
            fast_log(alpha_sum + count_sum);
    }

    void remove_value (
            const Model & model,
            size_t groupid,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(groupid < groups.size(), "groupid out of bounds");
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        Group & group = groups[groupid];
        count_t count_sum = group.count_sum -= 1;
        count_t count = group.counts[value] -= 1;
        scores[value][groupid] = fast_log(model.alphas[value] + count);
        scores_shift[groupid] =
            fast_log(alpha_sum + count_sum);
    }

    void score_value (
            const Model & model,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        DIST_ASSERT2(
            scores_accum.size() == groups.size(),
            "expected scores_accum.size() = " << groups.size() <<
            ", actual " << scores_accum.size());
        const size_t group_count = groups.size();
        vector_add_subtract(
            group_count,
            scores_accum.data(),
            scores[value].data(),
            scores_shift.data());
    }
};

class Fitter
{
    std::vector<Group> groups;
    VectorFloat scores;
};

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
    DIST_ASSERT1(value < dim, "value out of bounds: " << value);
    return fast_log(scorer.alphas[value] / scorer.alpha_sum);
}

float score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    DIST_ASSERT1(value < dim, "value out of bounds: " << value);
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
    DIST_ASSERT1(value < dim, "value out of bounds: " << value);

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

//----------------------------------------------------------------------------
// Examples

static DirichletDiscrete<max_dim> EXAMPLE ();

}; // struct DirichletDiscrete<max_dim>

template<int max_dim>
inline DirichletDiscrete<max_dim> DirichletDiscrete<max_dim>::EXAMPLE ()
{
    DirichletDiscrete<max_dim> model;
    model.dim = max_dim;
    for (int i = 0; i < max_dim; ++i) {
        model.alphas[i] = 0.5;
    }
    return model;
}

} // namespace distributions
