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
#include <distributions/mixins.hpp>
#include <distributions/mixture.hpp>

namespace distributions
{
struct BetaNegativeBinomial
{

typedef BetaNegativeBinomial Model;
typedef uint32_t Value;
struct Group;
struct Scorer;
struct Sampler;
struct VectorizedScorer;
typedef GroupScorerMixture<VectorizedScorer> Mixture;


struct Shared : SharedMixin<Model>
{
    float alpha;
    float beta;
    uint32_t r;

    Shared plus_group (const Group & group) const
    {
        Shared post;
        post.alpha = alpha + float(r) * group.count;
        post.beta = beta + group.sum;
        post.r = r;
        return post;
    }

    template<class Message>
    void protobuf_load (const Message & message)
    {
        alpha = message.alpha();
        beta = message.beta();
        r = message.r();
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.set_alpha(alpha);
        message.set_beta(beta);
        message.set_r(r);
    }

    static Shared EXAMPLE ()
    {
        Shared shared;
        shared.alpha = 1.0;
        shared.beta = 1.0;
        shared.r = 1U;
        return shared;
    }
};


struct Group : GroupMixin<Model>
{
    uint32_t count;
    uint32_t sum;

    template<class Message>
    void protobuf_load (const Message & message)
    {
        count = message.count();
        sum = message.sum();
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.set_count(count);
        message.set_sum(sum);
    }

    void init (const Shared &, rng_t &)
    {
        count = 0;
        sum = 0;
    }

    void add_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        ++count;
        sum += value;
    }

    void remove_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        --count;
        sum -= value;
    }

    void merge (
            const Shared &,
            const Group & source,
            rng_t &)
    {
        count += source.count;
        sum += source.sum;
    }

    float score_value (
            const Shared & shared,
            const Value & value,
            rng_t &) const
    {
        Shared post = shared.plus_group(*this);
        float alpha = post.alpha + shared.r;
        float beta = post.beta + value;
        float score = fast_lgamma(post.alpha + post.beta)
                    - fast_lgamma(alpha + beta);
        score += fast_lgamma(alpha) - fast_lgamma(post.alpha);
        score += fast_lgamma(beta) - fast_lgamma(post.beta);
        return score;
    }

    float score_data (
            const Shared & shared,
            rng_t &) const
    {
        Shared post = shared.plus_group(*this);
        float score = fast_lgamma(shared.alpha + shared.beta)
                    - fast_lgamma(post.alpha + post.beta);
        score += fast_lgamma(post.alpha) - fast_lgamma(shared.alpha);
        score += fast_lgamma(post.beta) - fast_lgamma(shared.beta);
        return score;
    }

    Value sample_value (
            const Shared & shared,
            rng_t & rng) const
    {
        Sampler sampler;
        sampler.init(shared, *this, rng);
        return sampler.eval(shared, rng);
    }
};

struct Sampler
{
    float beta;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        Shared post = shared.plus_group(group);
        beta = sample_beta(rng, post.alpha, post.beta);
    }

    Value eval (
            const Shared & shared,
            rng_t & rng) const
    {
        return sample_negative_binomial(rng, beta, shared.r);
    }
};

struct Scorer
{
    float score;
    float post_beta;
    float alpha;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        Shared post = shared.plus_group(group);
        post_beta = post.beta;
        alpha = post.alpha + shared.r;
        score = fast_lgamma(post.alpha + post.beta)
              - fast_lgamma(post.alpha)
              - fast_lgamma(post.beta)
              + fast_lgamma(alpha);
    }

    float eval (
            const Shared &,
            const Value & value,
            rng_t &) const
    {
        float beta = post_beta + value;
        return score + fast_lgamma(beta) - fast_lgamma(alpha + beta);
    }
};

struct VectorizedScorer : VectorizedScorerMixin<Model>
{
    void resize(const Shared &, size_t size)
    {
        score_.resize(size);
        post_beta_.resize(size);
        alpha_.resize(size);
    }

    void add_group (const Shared &, rng_t &)
    {
        score_.packed_add();
        post_beta_.packed_add();
        alpha_.packed_add();
    }

    void remove_group (const Shared &, size_t groupid)
    {
        score_.packed_remove(groupid);
        post_beta_.packed_remove(groupid);
        alpha_.packed_remove(groupid);
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng)
    {
        Model::Scorer base;
        base.init(shared, group, rng);

        score_[groupid] = base.score;
        post_beta_[groupid] = base.post_beta;
        alpha_[groupid] = base.alpha;
    }

    void add_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value &,
            rng_t & rng)
    {
        update_group(shared, groupid, group, rng);
    }

    void remove_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value &,
            rng_t & rng)
    {
        update_group(shared, groupid, group, rng);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t & rng)
    {
        const size_t group_count = slave.groups().size();
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            update_group(shared, groupid, slave.groups()[groupid], rng);
        }
    }

    void score_value (
            const Shared &,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        for (size_t i = 0, size = scores_accum.size(); i < size; ++i) {
            float beta = post_beta_[i] + value;
            scores_accum[i] += score_[i] + fast_lgamma(beta)
                                         - fast_lgamma(beta + alpha_[i]);
        }
    }

    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &) const
    {
        const float shared_part = fast_lgamma(shared.alpha + shared.beta)
                                - fast_lgamma(shared.alpha)
                                - fast_lgamma(shared.beta);
        float score = 0;
        for (const auto & group : slave.groups()) {
            if (group.count) {
                Shared post = shared.plus_group(group);
                score += fast_lgamma(post.alpha)
                       + fast_lgamma(post.beta)
                       - fast_lgamma(post.alpha + post.beta);
                score += shared_part;
            }
        }
        return score;
    }

    void score_data_grid (
            const std::vector<Shared> & shareds,
            const MixtureSlave<Shared> & slave,
            AlignedFloats scores_out,
            rng_t & rng) const
    {
        slave.score_data_grid(shareds, scores_out, rng);
    }

private:

    VectorFloat score_;
    VectorFloat post_beta_;
    VectorFloat alpha_;
};

}; // struct BetaNegativeBinomial
} // namespace distributions
