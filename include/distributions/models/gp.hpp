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
typedef GammaPoisson Model;

static const char * name () { return "GammaPoisson"; }
static const char * short_name () { return "gp"; }

//----------------------------------------------------------------------------
// Data

float alpha;
float inv_beta;

//----------------------------------------------------------------------------
// Datatypes

typedef uint32_t Value;

struct Group
{
    uint32_t count;
    uint32_t sum;
    float log_prod;

    void init (const Model &, rng_t &)
    {
        count = 0;
        sum = 0;
        log_prod = 0.f;
    }

    void add_value (
            const Model &,
            const Value & value,
            rng_t &)
    {
        ++count;
        sum += value;
        log_prod += fast_log_factorial(value);
    }

    void remove_value (
            const Model &,
            const Value & value,
            rng_t &)
    {
        --count;
        sum -= value;
        log_prod -= fast_log_factorial(value);
    }

    void merge (
            const Model &,
            const Group & source,
            rng_t &)
    {
        count += source.count;
        sum += source.sum;
        log_prod += source.log_prod;
    }
};

struct Sampler
{
    float mean;

    void init (
            const Model & model,
            const Group & group,
            rng_t & rng)
    {
        Model post = model.plus_group(group);
        mean = sample_gamma(rng, post.alpha, 1.f / post.inv_beta);
    }

    Value eval (
            const Model & model,
            rng_t & rng) const
    {
        return sample_poisson(rng, mean);
    }
};

struct Scorer
{
    float score;
    float post_alpha;
    float score_coeff;

    void init (
            const Model & model,
            const Group & group,
            rng_t &)
    {
        Model post = model.plus_group(group);
        score_coeff = -fast_log(1.f + post.inv_beta);
        score = -fast_lgamma(post.alpha)
                     + post.alpha * (fast_log(post.inv_beta) + score_coeff);
        post_alpha = post.alpha;
    }

    float eval (
            const Model & model,
            const Value & value,
            rng_t &) const
    {
        return score
             + fast_lgamma(post_alpha + value)
             - fast_log_factorial(value)
             + score_coeff * value;
    }
};

struct Classifier
{
    std::vector<Group> groups;
    VectorFloat score;
    VectorFloat post_alpha;
    VectorFloat score_coeff;
    mutable VectorFloat temp;

    private:

    void _update_group (
            const Model & model,
            size_t groupid,
            rng_t & rng)
    {
        const Group & group = groups[groupid];
        Scorer scorer;
        scorer.init(model, group, rng);
        score[groupid] = scorer.score;
        post_alpha[groupid] = scorer.post_alpha;
        score_coeff[groupid] = scorer.score_coeff;
    }

    void _resize (
            const Model & model,
            size_t group_count)
    {
        groups.resize(group_count);
        score.resize(group_count);
        post_alpha.resize(group_count);
        score_coeff.resize(group_count);
        temp.resize(group_count);
    }

    public:

    void init (
            const Model & model,
            rng_t & rng)
    {
        const size_t group_count = groups.size();
        _resize(model, group_count);
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            _update_group(model, groupid, rng);
        }
    }

    void add_group (
            const Model & model,
            rng_t & rng)
    {
        const size_t groupid = groups.size();
        const size_t group_count = groupid + 1;
        _resize(model, group_count);
        groups.back().init(model, rng);
        _update_group(model, groupid, rng);
    }

    void remove_group (
            const Model & model,
            size_t groupid)
    {
        const size_t group_count = groups.size() - 1;
        if (groupid != group_count) {
            std::swap(groups[groupid], groups.back());
            score[groupid] = score.back();
            post_alpha[groupid] = post_alpha.back();
            score_coeff[groupid] = score_coeff.back();
        }
        _resize(model, group_count);
    }

    void add_value (
            const Model & model,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT1(groupid < groups.size(), "groupid out of bounds");
        Group & group = groups[groupid];
        group.add_value(model, value, rng);
        _update_group(model, groupid, rng);
    }

    void remove_value (
            const Model & model,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT1(groupid < groups.size(), "groupid out of bounds");
        Group & group = groups[groupid];
        group.remove_value(model, value, rng);
        _update_group(model, groupid, rng);
    }

    void score_value (
            const Model & model,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const;
};

//----------------------------------------------------------------------------
// Mutation


Model plus_group (const Group & group) const
{
    Model post;
    post.alpha = alpha + group.sum;
    post.inv_beta = inv_beta + group.count;
    return post;
}

Value sample_value (
        const Group & group,
        rng_t & rng) const
{
    Sampler sampler;
    sampler.init(*this, group, rng);
    return sampler.eval(*this, rng);
}

float score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    Scorer scorer;
    scorer.init(*this, group, rng);
    return scorer.eval(*this, value, rng);
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

//----------------------------------------------------------------------------
// Examples

static GammaPoisson EXAMPLE ();

}; // struct GammaPoisson

inline GammaPoisson GammaPoisson::EXAMPLE ()
{
    GammaPoisson model;
    model.alpha = 1.0;
    model.inv_beta = 1.0;
    return model;
}

} // namespace distributions
