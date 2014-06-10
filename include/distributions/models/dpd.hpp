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
#include <distributions/sparse.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/mixins.hpp>
#include <distributions/mixture.hpp>

namespace distributions
{
struct DirichletProcessDiscrete
{

typedef DirichletProcessDiscrete Model;
typedef uint32_t count_t;
typedef uint32_t Value;
struct Group;
struct Scorer;
struct Sampler;
struct VectorizedScorer;
typedef GroupScorerMixture<VectorizedScorer> Mixture;

struct Shared : SharedMixin<Model>
{
    float gamma;
    float alpha;
    float beta0;
    Sparse_<Value, float> betas;
    SparseCounter<Value, count_t> counts;

    static constexpr Value OTHER () { return 0xFFFFFFFFU; }

    void add_value (const Value & value, rng_t & rng)
    {
        DIST_ASSERT1(value != OTHER(), "cannot add OTHER");
        if (DIST_UNLIKELY(counts.add(value) == 1)) {
            float beta = beta0 * sample_beta(rng, 1.f, gamma);
            beta0 = std::max(0.f, beta0 - beta);
            betas.add(value, beta);
        }
    }

    void remove_value (const Value & value, rng_t &)
    {
        DIST_ASSERT1(value != OTHER(), "cannot remove OTHER");
        if (DIST_UNLIKELY(counts.remove(value) == 0)) {
            beta0 += betas.pop(value);
        }
    }

    void realize (rng_t & rng)
    {
        const size_t max_size = 10000;
        const float min_beta0 = 1e-4f;

        Value new_value = 0;
        for (const auto & i : betas) {
            new_value = std::max(new_value, 1 + i.first);
        }

        while (betas.size() < max_size - 1 and beta0 > min_beta0) {
            add_value(new_value++, rng);
        }

        if (beta0 > 0) {
            add_value(new_value, rng);
            betas.get(new_value) += beta0;
            beta0 = 0;
        }
    }

    template<class Message>
    void protobuf_load (const Message & message)
    {
        const size_t value_count = message.values_size();
        DIST_ASSERT_EQ(message.betas_size(), value_count);
        DIST_ASSERT_EQ(message.counts_size(), value_count);
        gamma = message.gamma();
        alpha = message.alpha();
        betas.clear();
        counts.clear();
        double beta_sum = 0;
        for (size_t i = 0; i < value_count; ++i) {
            auto value = message.values(i);
            float beta = message.betas(i);
            DIST_ASSERT_LT(0, beta);
            betas.add(value, beta);
            beta_sum += beta;
            counts.add(value, message.counts(i));
        }
        DIST_ASSERT_LE(beta_sum, 1 + 1e-4);
        beta0 = std::max(0.0, 1.0 - beta_sum);
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.Clear();
        message.set_gamma(gamma);
        message.set_alpha(alpha);
        for (auto & i : betas) {
            auto value = i.first;
            auto beta = i.second;
            message.add_values(value);
            message.add_betas(beta);
            message.add_counts(counts.get_count(value));
        }
    }

    static Shared EXAMPLE ()
    {
        Shared shared;
        size_t dim = 100;
        shared.gamma = 1.0 / dim;
        shared.alpha = 0.5;
        shared.beta0 = 0.0;  // must be zero for testing
        for (size_t i = 0; i < dim; ++i) {
            shared.betas.add(i, 1.0 / dim);
            shared.counts.add(i);
        }
        return shared;
    }
};


struct Group : GroupMixin<Model>
{
    SparseCounter<Value, count_t> counts;

    template<class Message>
    void protobuf_load (const Message & message)
    {
        if (DIST_DEBUG_LEVEL >= 1) {
            DIST_ASSERT_EQ(message.keys_size(), message.values_size());
        }
        counts.clear();
        for (size_t i = 0, size = message.keys_size(); i < size; ++i) {
            counts.add(message.keys(i), message.values(i));
        }
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.Clear();
        auto & keys = * message.mutable_keys();
        auto & values = * message.mutable_values();
        for (const auto & pair : counts) {
            keys.Add(pair.first);
            values.Add(pair.second);
        }
    }

    void init (
            const Shared &,
            rng_t &)
    {
        counts.clear();
    }

    void add_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value != shared.OTHER(), "cannot add OTHER");
        DIST_ASSERT1(shared.betas.contains(value), "unknown value: " << value);
        counts.add(value);
    }

    void remove_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value != shared.OTHER(), "cannot remove OTHER");
        DIST_ASSERT1(shared.betas.contains(value), "unknown value: " << value);
        counts.remove(value);
    }

    void merge (
            const Shared &,
            const Group & source,
            rng_t &)
    {
        counts.merge(source.counts);
    }

    float score_value (
            const Shared & shared,
            const Value & value,
            rng_t &) const
    {
        float alpha = shared.alpha;
        float numer = (value == shared.OTHER())
                    ? alpha * shared.beta0
                    : alpha * shared.betas.get(value) + counts.get_count(value);
        float denom = alpha + counts.get_total();
        return fast_log(numer / denom);
    }

    float score_data (
            const Shared & shared,
            rng_t &) const
    {
        const size_t total = counts.get_total();
        const float alpha = shared.alpha;

        float score = 0;
        for (auto & i : counts) {
            Value value = i.first;
            DIST_ASSERT3(
                shared.betas.contains(value),
                "shared.betas is missing value: " << value);
            float prior_i = alpha * shared.betas.get(value);
            score += fast_lgamma(prior_i + i.second)
                   - fast_lgamma(prior_i);
        }
        score += fast_lgamma(alpha)
               - fast_lgamma(alpha + total);

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
    std::vector<float> probs;
    std::vector<Value> values;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        probs.clear();
        probs.reserve(shared.betas.size() + 1);
        values.clear();
        values.reserve(shared.betas.size() + 1);
        const float alpha = shared.alpha;
        for (auto & pair : shared.betas) {
            Value value = pair.first;
            float beta = pair.second;
            values.push_back(value);
            probs.push_back(beta * alpha + group.counts.get_count(value));
        }
        if (shared.beta0 > 0) {
            values.push_back(shared.OTHER());
            probs.push_back(shared.beta0 * alpha);
        }

        sample_dirichlet(rng, probs.size(), probs.data(), probs.data());
    }

    Value eval (
            const Shared &,
            rng_t & rng) const
    {
        size_t index = sample_discrete(rng, probs.size(), probs.data());
        return values[index];
    }
};

struct Scorer
{
    Sparse_<Value, float> scores;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        scores.clear();

        const size_t total = group.counts.get_total();
        const float beta_scale = shared.alpha / (shared.alpha + total);
        scores.add(shared.OTHER(), beta_scale * shared.beta0);
        for (auto & i : shared.betas) {
            scores.add(i.first, i.second * beta_scale);
        }

        const float counts_scale = 1.0f / (shared.alpha + total);
        for (auto & i : group.counts) {
            scores.get(i.first) += counts_scale * i.second;
        }

        for (auto & i : scores) {
            float & score = i.second;
            score = fast_log(score);
        }
    }

    float eval (
            const Shared &,
            const Value & value,
            rng_t &) const
    {
        return scores.get(value);
    }
};

struct VectorizedScorer : VectorizedScorerMixin<Model>
{
    void resize (const Shared & shared, size_t size)
    {
        scores_shift_.resize(size);
        for (const auto & i : shared.betas) {
            Value value = i.first;
            scores_.get_or_add(value).resize(size);
        }
        if (scores_.size() != shared.betas.size()) {
            for (auto i = scores_.begin(); i != scores_.end();) {
                if (DIST_UNLIKELY(not shared.betas.contains(i->first))) {
                    scores_.unsafe_erase(i++);
                } else {
                    ++i;
                }
            }
        }
    }

    void add_group (const Shared & shared, rng_t &)
    {
        const float alpha = shared.alpha;
        for (auto & i : scores_) {
            i.second.packed_add(fast_log(alpha * shared.betas.get(i.first)));
        }
        scores_shift_.packed_add(fast_log(alpha));
    }

    void remove_group (const Shared &, size_t groupid)
    {
        for (auto & i : scores_) {
            i.second.packed_remove(groupid);
        }
        scores_shift_.packed_remove(groupid);
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t &)
    {
        const float alpha = shared.alpha;
        for (auto & i : scores_) {
            Value value = i.first;
            count_t count = group.counts.get_count(value);
            i.second[groupid] =
                fast_log(alpha * shared.betas.get(value) + count);
        }
        count_t count_sum = group.counts.get_total();
        scores_shift_[groupid] = fast_log(alpha + count_sum);
    }

    void add_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value != shared.OTHER(), "cannot add OTHER");
        if (DIST_UNLIKELY(not scores_.contains(value))) {
            const size_t group_count = scores_shift_.size();
            const float beta = shared.alpha * shared.betas.get(value);
            scores_.add(value).resize(group_count, fast_log(beta));
        }

        _update_group_value(shared, groupid, group, value);
    }

    void remove_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value != shared.OTHER(), "cannot remove OTHER");
        if (DIST_UNLIKELY(not shared.betas.contains(value))) {
            scores_.remove(value);
        }

        _update_group_value(shared, groupid, group, value);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &)
    {
        const size_t group_count = slave.groups().size();
        const float alpha = shared.alpha;

        for (auto & i : scores_) {
            Value value = i.first;
            VectorFloat & scores = i.second;
            const float beta = shared.betas.get(value);
            for (size_t groupid = 0; groupid < group_count; ++groupid) {
                auto count = slave.groups(groupid).counts.get_count(value);
                scores[groupid] = alpha * beta + count;
            }
            vector_log(group_count, scores.data());
        }

        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            auto total = slave.groups(groupid).counts.get_total();
            scores_shift_[groupid] = alpha + total;
        }
        vector_log(group_count, scores_shift_.data());
    }

    void score_value (
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        if (DIST_LIKELY(scores_.contains(value))) {

            vector_add_subtract(
                scores_accum.size(),
                scores_accum.data(),
                scores_.get(value).data(),
                scores_shift_.data());

        } else {

            float beta = (value == shared.OTHER())
                       ? shared.beta0
                       : shared.betas.get(value);
            float score = fast_log(shared.alpha * beta);
            vector_add_subtract(
                scores_accum.size(),
                scores_accum.data(),
                score,
                scores_shift_.data());
        }
    }

    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &) const
    {
        const float alpha = shared.alpha;

        Sparse_<Value, float> shared_part;
        for (auto & i : shared.betas) {
            shared_part.add(i.first, fast_lgamma(alpha * i.second));
        }
        const float shared_total = fast_lgamma(alpha);

        float score = 0;
        for (const auto & group : slave.groups()) {
            if (group.counts.get_total()) {
                for (auto & i : group.counts) {
                    Value value = i.first;
                    float prior_i = shared.betas.get(value) * alpha;
                    score += fast_lgamma(prior_i + i.second)
                           - shared_part.get(value);
                }
                score += shared_total
                       - fast_lgamma(alpha + group.counts.get_total());
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

    void _update_group_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value)
    {
        const float alpha = shared.alpha;
        count_t count = group.counts.get_count(value);
        count_t count_sum = group.counts.get_total();
        scores_.get(value)[groupid] =
            fast_log(alpha * shared.betas.get(value) + count);
        scores_shift_[groupid] = fast_log(alpha + count_sum);
    }

    Sparse_<Value, VectorFloat> scores_;
    VectorFloat scores_shift_;
};

}; // struct DirichletProcessDiscrete
} // namespace distributions
