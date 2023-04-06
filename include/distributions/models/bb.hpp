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

#include <vector>
#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/mixins.hpp>
#include <distributions/mixture.hpp>

namespace distributions {
struct BetaBernoulli {
typedef BetaBernoulli Model;
typedef int count_t;
typedef bool Value;
struct Group;
struct Scorer;
struct Sampler;
struct MixtureDataScorer;
struct MixtureValueScorer;
typedef MixtureSlave<Model, MixtureDataScorer> SmallMixture;
typedef MixtureSlave<Model, MixtureDataScorer, MixtureValueScorer> FastMixture;
typedef FastMixture Mixture;


struct Shared : SharedMixin<Model> {
    float alpha;
    float beta;

    template<class Message>
    void protobuf_load(const Message & message) {
        alpha = message.alpha();
        beta = message.beta();
    }

    template<class Message>
    void protobuf_dump(Message & message) const {
        message.set_alpha(alpha);
        message.set_beta(beta);
    }

    static Shared EXAMPLE() {
        Shared shared;
        shared.alpha = 0.5;
        shared.beta = 2.0;
        return shared;
    }
};


struct Group : GroupMixin<Model> {
    count_t heads;
    count_t tails;

    template<class Message>
    void protobuf_load(const Message & message) {
        heads = message.heads();
        tails = message.tails();
    }

    template<class Message>
    void protobuf_dump(Message & message) const {
        message.set_heads(heads);
        message.set_tails(tails);
    }

    void init(
            const Shared &,
            rng_t &) {
        heads = 0;
        tails = 0;
    }

    void add_value(
            const Shared &,
            const Value & value,
            rng_t &) {
        (value ? heads : tails) += 1;
    }

    void add_repeated_value(
            const Shared &,
            const Value & value,
            const int & count,
            rng_t &) {
        (value ? heads : tails) += count;
    }

    void remove_value(
            const Shared &,
            const Value & value,
            rng_t &) {
        (value ? heads : tails) -= 1;
    }

    void merge(
            const Shared &,
            const Group & source,
            rng_t &) {
        heads += source.heads;
        tails += source.tails;
    }

    float score_value(
            const Shared & shared,
            const Value & value,
            rng_t & rng) const {
        Scorer scorer;
        scorer.init(shared, *this, rng);
        return scorer.eval(shared, value, rng);
    }

    float score_data(
            const Shared & shared,
            rng_t &) const {
        float alpha = shared.alpha + heads;
        float beta = shared.beta + tails;
        float score = 0;
        score += fast_lgamma(alpha) - fast_lgamma(shared.alpha);
        score += fast_lgamma(beta) - fast_lgamma(shared.beta);
        score += fast_lgamma(shared.alpha + shared.beta)
               - fast_lgamma(alpha + beta);
        return score;
    }

    Value sample_value(
            const Shared & shared,
            rng_t & rng) const {
        Sampler sampler;
        sampler.init(shared, *this, rng);
        return sampler.eval(shared, rng);
    }
};

struct Sampler {
    float heads_prob;

    void init(
            const Shared & shared,
            const Group & group,
            rng_t & rng) {
        float ps[2] = {
            shared.alpha + group.heads,
            shared.beta + group.tails
        };
        sample_dirichlet(rng, 2, ps, ps);
        heads_prob = ps[0];
    }

    Value eval(
            const Shared &,
            rng_t & rng) const {
        return sample_bernoulli(rng, heads_prob);
    }
};

struct Scorer {
    float heads_score;
    float tails_score;

    void init(
            const Shared & shared,
            const Group & group,
            rng_t &) {
        float alpha = shared.alpha + group.heads;
        float beta = shared.beta + group.tails;
        heads_score = fast_log(alpha / (alpha + beta));
        tails_score = fast_log(beta / (alpha + beta));
    }

    float eval(
            const Shared &,
            const Value & value,
            rng_t &) const {
        return value ? heads_score : tails_score;
    }
};

struct MixtureDataScorer
    : MixtureSlaveDataScorerMixin<Model, MixtureDataScorer> {
    float score_data(
            const Shared & shared,
            const std::vector<Group> & groups,
            rng_t &) const {
        const float shared_part =
               + fast_lgamma(shared.alpha + shared.beta)
               - fast_lgamma(shared.alpha)
               - fast_lgamma(shared.beta);
        float score = 0;
        for (auto const & group : groups) {
            float alpha = shared.alpha + group.heads;
            float beta = shared.beta + group.tails;
            float group_part =
                   + fast_lgamma(alpha)
                   + fast_lgamma(beta)
                   - fast_lgamma(alpha + beta);
            score += shared_part + group_part;
        }
        return score;
    }
};

struct MixtureValueScorer : MixtureSlaveValueScorerMixin<Model> {
    void resize(const Shared &, size_t size) {
        heads_scores_.resize(size);
        tails_scores_.resize(size);
    }

    void add_group(const Shared &, rng_t &) {
        heads_scores_.packed_add(0);
        tails_scores_.packed_add(0);
    }

    void remove_group(const Shared &, size_t groupid) {
        heads_scores_.packed_remove(groupid);
        tails_scores_.packed_remove(groupid);
    }

    void update_group(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng) {
        Scorer scorer;
        scorer.init(shared, group, rng);
        heads_scores_[groupid] = scorer.heads_score;
        tails_scores_[groupid] = scorer.tails_score;
    }

    void add_value(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value &,
            rng_t & rng) {
        update_group(shared, groupid, group, rng);
    }

    void remove_value(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value &,
            rng_t & rng) {
        update_group(shared, groupid, group, rng);
    }

    void update_all(
            const Shared & shared,
            const std::vector<Group> & groups,
            rng_t &) {
        const size_t group_count = groups.size();
        heads_scores_.resize(group_count);
        tails_scores_.resize(group_count);
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = groups[groupid];
            float heads = shared.alpha + group.heads;
            float tails = shared.beta + group.tails;
            heads_scores_[groupid] = heads / (heads + tails);
            tails_scores_[groupid] = tails / (heads + tails);
        }
        vector_log(group_count, heads_scores_.data());
        vector_log(group_count, tails_scores_.data());
    }

    float score_value_group(
            const Shared &,
            const std::vector<Group> &,
            size_t groupid,
            const Value & value,
            rng_t &) const {
        return value ? heads_scores_[groupid] : tails_scores_[groupid];
    }

    void score_value(
            const Shared &,
            const std::vector<Group> &,
            const Value & value,
            AlignedFloats scores_accum,
            rng_t &) const {
        vector_add(
            scores_accum.size(),
            scores_accum.data(),
            (value ? heads_scores_.data() : tails_scores_.data()));
    }

    void validate(
            const Shared &,
            const std::vector<Group> & groups) const {
        DIST_ASSERT_EQ(heads_scores_.size(), groups.size());
        DIST_ASSERT_EQ(tails_scores_.size(), groups.size());
    }

 private:
    VectorFloat heads_scores_;
    VectorFloat tails_scores_;
};
};  // struct BetaBernoulli
}   // namespace distributions
