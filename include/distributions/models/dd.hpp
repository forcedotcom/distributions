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

namespace distributions {
namespace dirichlet_discrete {
typedef uint32_t count_t;
typedef int Value;

template<int max_dim> struct Group;
template<int max_dim> struct Scorer;
template<int max_dim> struct Sampler;
template<int max_dim> struct Mixture;

template<int max_dim>
struct Model
{
typedef dirichlet_discrete::Value Value;
typedef typename dirichlet_discrete::Group<max_dim> Group;
typedef typename dirichlet_discrete::Scorer<max_dim> Scorer;
typedef typename dirichlet_discrete::Sampler<max_dim> Sampler;
typedef typename dirichlet_discrete::Mixture<max_dim> Mixture;

static const char * name () { return "DirichletDiscrete"; }
static const char * short_name () { return "dd"; }

int dim;  // fixed parameter
float alphas[max_dim];  // hyperparamter

Value sample_value(const Group & group, rng_t & rng) const;
float score_value(const Group & group, const Value & value, rng_t & rng) const;
float score_group(const Group & group, rng_t &) const;

static Model<max_dim> EXAMPLE ();

}; // struct DirichletDiscrete<max_dim>

template<int max_dim>
inline Model<max_dim> Model<max_dim>::EXAMPLE ()
{
    Model<max_dim> model;
    model.dim = max_dim;
    for (int i = 0; i < max_dim; ++i) {
        model.alphas[i] = 0.5;
    }
    return model;
}

template<int max_dim>
struct Group
{
    count_t count_sum;
    count_t counts[max_dim];

    void init (
            const Model<max_dim> & model,
            rng_t &)
    {
        count_sum = 0;
        for (Value value = 0; value < model.dim; ++value) {
            counts[value] = 0;
        }
    }

    void add_value (
            const Model<max_dim> & model,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < model.dim, "bad value: out of bounds: " << value);
        count_sum += 1;
        counts[value] += 1;
    }

    void remove_value (
            const Model<max_dim> & model,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        count_sum -= 1;
        counts[value] -= 1;
    }

    void merge (
            const Model<max_dim> & model,
            const Group<max_dim> & source,
            rng_t &)
    {
        for (Value value = 0; value < model.dim; ++value) {
            counts[value] += source.counts[value];
        }
    }
};

template<int max_dim>
struct Sampler
{
    float ps[max_dim];

    void init (
            const Model<max_dim> & model,
            const Group<max_dim> & group,
            rng_t & rng)
    {
        for (Value value = 0; value < model.dim; ++value) {
            ps[value] = model.alphas[value] + group.counts[value];
        }

        sample_dirichlet(rng, model.dim, ps, ps);
    }

    Value eval (
            const Model<max_dim> & model,
            rng_t & rng) const
    {
        return sample_discrete(rng, model.dim, ps);
    }
};

template<int max_dim>
struct Scorer
{
    float alpha_sum;
    float alphas[max_dim];

    void init (
            const Model<max_dim> & model,
            const Group<max_dim> & group,
            rng_t &)
    {
        alpha_sum = 0;
        for (Value value = 0; value < model.dim; ++value) {
            float alpha = model.alphas[value] + group.counts[value];
            alphas[value] = alpha;
            alpha_sum += alpha;
        }
    }

    float eval (
            const Model<max_dim> & model,
            const Value & value,
            rng_t &) const
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        return fast_log(alphas[value] / alpha_sum);
    }
};

template<int max_dim>
struct Mixture
{
    std::vector<Group<max_dim>> groups;
    float alpha_sum;
    std::vector<VectorFloat> scores;
    VectorFloat scores_shift;

    void init (
            const Model<max_dim> & model,
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
            const Group<max_dim> & group = groups[groupid];
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
            const Model<max_dim> & model,
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
            const Model<max_dim> & model,
            size_t groupid)
    {
        DIST_ASSERT1(groupid < groups.size(), "bad groupid: " << groupid);
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
            const Model<max_dim> & model,
            size_t groupid,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(groupid < groups.size(), "bad groupid: " << groupid);
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        Group<max_dim> & group = groups[groupid];
        count_t count_sum = group.count_sum += 1;
        count_t count = group.counts[value] += 1;
        scores[value][groupid] = fast_log(model.alphas[value] + count);
        scores_shift[groupid] =
            fast_log(alpha_sum + count_sum);
    }

    void remove_value (
            const Model<max_dim> & model,
            size_t groupid,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT2(groupid < groups.size(), "bad groupid: " << groupid);
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        Group<max_dim> & group = groups[groupid];
        count_t count_sum = group.count_sum -= 1;
        count_t count = group.counts[value] -= 1;
        scores[value][groupid] = fast_log(model.alphas[value] + count);
        scores_shift[groupid] =
            fast_log(alpha_sum + count_sum);
    }

    void score_value (
            const Model<max_dim> & model,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        DIST_ASSERT1(value < model.dim, "value out of bounds: " << value);
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_EQ(scores_accum.size(), groups.size());
        }
        const size_t group_count = groups.size();
        vector_add_subtract(
            group_count,
            scores_accum.data(),
            scores[value].data(),
            scores_shift.data());
    }
};

template<int max_dim>
Value Model<max_dim>::sample_value (
        const Group & group,
        rng_t & rng) const
{
    Sampler sampler;
    sampler.init(*this, group, rng);
    return sampler.eval(*this, rng);
}

template<int max_dim>
float Model<max_dim>::score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    Scorer scorer;
    scorer.init(*this, group, rng);
    return scorer.eval(*this, value, rng);
}

template<int max_dim>
float Model<max_dim>::score_group (
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



} // namespace dirichlet_discrete

// g++4.6 doesn't support template aliases, so use this ugly hack
template <int max_dim>
struct DirichletDiscrete : public dirichlet_discrete::Model<max_dim>
{};

} // namespace distributions
