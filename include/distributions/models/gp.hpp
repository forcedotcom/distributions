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
#include <distributions/mixture.hpp>

namespace distributions {
namespace gamma_poisson {

typedef uint32_t Value;
struct Group;
struct Scorer;
struct Sampler;
struct Mixture;


struct Shared
{
    typedef gamma_poisson::Value Value;
    typedef gamma_poisson::Group Group;

    float alpha;
    float inv_beta;

    Shared plus_group(const Group & group) const;

    static Shared EXAMPLE ()
    {
        Shared shared;
        shared.alpha = 1.0;
        shared.inv_beta = 1.0;
        return shared;
    }
};


struct Group
{
    typedef gamma_poisson::Value Value;

    uint32_t count;
    uint32_t sum;
    float log_prod;

    void init (const Shared &, rng_t &)
    {
        count = 0;
        sum = 0;
        log_prod = 0.f;
    }

    void add_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        ++count;
        sum += value;
        log_prod += fast_log_factorial(value);
    }

    void remove_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        --count;
        sum -= value;
        log_prod -= fast_log_factorial(value);
    }

    void merge (
            const Shared &,
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
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        Shared post = shared.plus_group(group);
        mean = sample_gamma(rng, post.alpha, 1.f / post.inv_beta);
    }

    Value eval (
            const Shared & shared,
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
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        Shared post = shared.plus_group(group);
        score_coeff = -fast_log(1.f + post.inv_beta);
        score = -fast_lgamma(post.alpha)
                     + post.alpha * (fast_log(post.inv_beta) + score_coeff);
        post_alpha = post.alpha;
    }

    float eval (
            const Shared & shared,
            const Value & value,
            rng_t &) const
    {
        return score
             + fast_lgamma(post_alpha + value)
             - fast_log_factorial(value)
             + score_coeff * value;
    }
};

class Mixture
{
public:

    typedef gamma_poisson::Value Value;
    typedef gamma_poisson::Shared Shared;
    typedef gamma_poisson::Group Group;
    typedef gamma_poisson::Scorer Scorer;

    std::vector<Group> & groups () { return groups_; }
    const std::vector<Group> & groups () const { return groups_; }

    void init (
            const Shared & shared,
            rng_t & rng)
    {
        const size_t group_count = groups_.size();
        _resize(shared, group_count);
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            _update_group(shared, groupid, rng);
        }
    }

    void add_group (
            const Shared & shared,
            rng_t & rng)
    {
        const size_t groupid = groups_.size();
        const size_t group_count = groupid + 1;
        _resize(shared, group_count);
        groups_.back().init(shared, rng);
        _update_group(shared, groupid, rng);
    }

    void remove_group (
            const Shared & shared,
            size_t groupid)
    {
        DIST_ASSERT1(groupid < groups_.size(), "bad groupid: " << groupid);
        const size_t group_count = groups_.size() - 1;
        if (groupid != group_count) {
            std::swap(groups_[groupid], groups_.back());
            score[groupid] = score.back();
            post_alpha[groupid] = post_alpha.back();
            score_coeff[groupid] = score_coeff.back();
        }
        _resize(shared, group_count);
    }

    void add_value (
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT1(groupid < groups_.size(), "bad groupid: " << groupid);
        Group & group = groups_[groupid];
        group.add_value(shared, value, rng);
        _update_group(shared, groupid, rng);
    }

    void remove_value (
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT2(groupid < groups_.size(), "bad groupid: " << groupid);
        Group & group = groups_[groupid];
        group.remove_value(shared, value, rng);
        _update_group(shared, groupid, rng);
    }

    void score_value (
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t & rng) const
    {
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_EQ(scores_accum.size(), groups_.size());
        }
        _score_value(shared, value, scores_accum, rng);
    }

private:

    void _update_group (
            const Shared & shared,
            size_t groupid,
            rng_t & rng)
    {
        const Group & group = groups_[groupid];
        Scorer scorer;
        scorer.init(shared, group, rng);
        score[groupid] = scorer.score;
        post_alpha[groupid] = scorer.post_alpha;
        score_coeff[groupid] = scorer.score_coeff;
    }

    void _resize (
            const Shared & shared,
            size_t group_count)
    {
        groups_.resize(group_count);
        score.resize(group_count);
        post_alpha.resize(group_count);
        score_coeff.resize(group_count);
        temp.resize(group_count);
    }

    void _score_value (
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const;

    std::vector<Group> groups_;
    VectorFloat score;
    VectorFloat post_alpha;
    VectorFloat score_coeff;
    mutable VectorFloat temp;
};

inline Shared Shared::plus_group (const Group & group) const
{
    Shared post;
    post.alpha = alpha + group.sum;
    post.inv_beta = inv_beta + group.count;
    return post;
}

inline Value sample_value (
        const Shared & shared,
        const Group & group,
        rng_t & rng)
{
    Sampler sampler;
    sampler.init(shared, group, rng);
    return sampler.eval(shared, rng);
}

inline float score_value (
        const Shared & shared,
        const Group & group,
        const Value & value,
        rng_t & rng)
{
    Scorer scorer;
    scorer.init(shared, group, rng);
    return scorer.eval(shared, value, rng);
}

inline float score_group (
        const Shared & shared,
        const Group & group,
        rng_t &)
{
    Shared post = shared.plus_group(group);
    float score = fast_lgamma(post.alpha) - fast_lgamma(shared.alpha);
    score += shared.alpha * fast_log(shared.inv_beta) - post.alpha * fast_log(post.inv_beta);
    score += -group.log_prod;
    return score;
}

} // namespace gamma_poisson
} // namespace distributions
