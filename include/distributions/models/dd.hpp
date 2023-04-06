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
template<int max_dim_>
struct DirichletDiscrete {
enum { max_dim = max_dim_ };

typedef DirichletDiscrete<max_dim> Model;
typedef int count_t;
typedef int Value;
struct Group;
struct Scorer;
struct Sampler;
struct MixtureDataScorer;
struct MixtureValueScorer;
typedef MixtureSlave<Model, MixtureDataScorer> SmallMixture;
typedef MixtureSlave<Model, MixtureDataScorer, MixtureValueScorer> FastMixture;
typedef FastMixture Mixture;


struct Shared : SharedMixin<Model> {
    int dim;  // fixed parameter
    float alphas[max_dim];  // hyperparamter

    template<class Message>
    void protobuf_load(const Message & message) {
        dim = message.alphas_size();
        DIST_ASSERT_LE(dim, max_dim);
        for (int i = 0; i < dim; ++i) {
            alphas[i] = message.alphas(i);
        }
    }

    template<class Message>
    void protobuf_dump(Message & message) const {
        message.Clear();
        for (int i = 0; i < dim; ++i) {
            message.add_alphas(alphas[i]);
        }
    }

    static Shared EXAMPLE() {
        Shared shared;
        shared.dim = max_dim;
        for (int i = 0; i < max_dim; ++i) {
            shared.alphas[i] = 0.5;
        }
        return shared;
    }
};


struct Group : GroupMixin<Model> {
    int dim;
    count_t count_sum;
    count_t counts[max_dim];

    template<class Message>
    void protobuf_load(const Message & message) {
        dim = message.counts_size();
        DIST_ASSERT_LE(dim, max_dim);
        count_sum = 0;
        for (int i = 0; i < dim; ++i) {
            count_sum += counts[i] = message.counts(i);
        }
    }

    template<class Message>
    void protobuf_dump(Message & message) const {
        message.Clear();
        auto & message_counts = * message.mutable_counts();
        for (int i = 0; i < dim; ++i) {
            message_counts.Add(counts[i]);
        }
    }

    void init(
            const Shared & shared,
            rng_t &) {
        dim = shared.dim;
        count_sum = 0;
        for (Value value = 0; value < dim; ++value) {
            counts[value] = 0;
        }
    }

    void add_value(
            const Shared &,
            const Value & value,
            rng_t &) {
        DIST_ASSERT1(value < dim, "value out of bounds: " << value);
        count_sum += 1;
        counts[value] += 1;
    }

    void add_repeated_value(
            const Shared &,
            const Value & value,
            const int & count,
            rng_t &) {
        DIST_ASSERT1(value < dim, "value out of bounds: " << value);
        count_sum += count;
        counts[value] += count;
    }

    void remove_value(
            const Shared &,
            const Value & value,
            rng_t &) {
        DIST_ASSERT1(value < dim, "value out of bounds: " << value);
        count_sum -= 1;
        counts[value] -= 1;
    }

    void merge(
            const Shared &,
            const Group & source,
            rng_t &) {
        for (Value value = 0; value < dim; ++value) {
            counts[value] += source.counts[value];
        }
    }

    float score_value(
            const Shared & shared,
            const Value & value,
            rng_t & rng) const {
        Scorer scorer;
        scorer.init(shared, * this, rng);
        return scorer.eval(shared, value, rng);
    }

    float score_data(
            const Shared & shared,
            rng_t &) const {
        float score = 0;
        float alpha_sum = 0;

        for (Value value = 0; value < dim; ++value) {
            float alpha = shared.alphas[value];
            alpha_sum += alpha;
            score += fast_lgamma(alpha + counts[value])
                   - fast_lgamma(alpha);
        }

        score += fast_lgamma(alpha_sum)
               - fast_lgamma(alpha_sum + count_sum);

        return score;
    }

    Value sample_value(
            const Shared & shared,
            rng_t & rng) const {
        Sampler sampler;
        sampler.init(shared, *this, rng);
        return sampler.eval(shared, rng);
    }

    void validate(const Shared & shared) const {
        DIST_ASSERT_EQ(dim, shared.dim);
    }
};

struct Sampler {
    float ps[max_dim];

    void init(
            const Shared & shared,
            const Group & group,
            rng_t & rng) {
        for (Value value = 0; value < shared.dim; ++value) {
            ps[value] = shared.alphas[value] + group.counts[value];
        }

        sample_dirichlet(rng, shared.dim, ps, ps);
    }

    Value eval(
            const Shared & shared,
            rng_t & rng) const {
        return sample_discrete(rng, shared.dim, ps);
    }
};

struct Scorer {
    float alpha_sum;
    float alphas[max_dim];

    void init(
            const Shared & shared,
            const Group & group,
            rng_t &) {
        alpha_sum = 0;
        for (Value value = 0; value < shared.dim; ++value) {
            float alpha = shared.alphas[value] + group.counts[value];
            alphas[value] = alpha;
            alpha_sum += alpha;
        }
    }

    float eval(
            const Shared & shared,
            const Value & value,
            rng_t &) const {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        return fast_log(alphas[value] / alpha_sum);
    }
};

struct MixtureDataScorer
    : MixtureSlaveDataScorerMixin<Model, MixtureDataScorer> {
    // not thread safe
    float score_data(
            const Shared & shared,
            const std::vector<Group> & groups,
            rng_t &) const {
        _init(shared, groups);
        return _eval();
    }

    // not thread safe
    void score_data_grid(
            const std::vector<Shared> & shareds,
            const std::vector<Group> & groups,
            AlignedFloats scores_out,
            rng_t &) const {
        DIST_ASSERT_EQ(shareds.size(), scores_out.size());
        if (const size_t size = shareds.size()) {
            const int dim = shareds[0].dim;

            _init(shareds[0], groups);
            scores_out[0] = _eval();

            for (size_t i = 1; i < size; ++i) {
                const float * old_alphas = shareds[i-1].alphas;
                const float * new_alphas = shareds[i].alphas;
                for (Value value = 0; value < dim; ++value) {
                    const float & old_alpha = old_alphas[value];
                    const float & new_alpha = new_alphas[value];
                    if (DIST_UNLIKELY(new_alpha != old_alpha)) {
                        _update(value, old_alpha, new_alpha, groups);
                    }
                }
                scores_out[i] = _eval();
            }
        }
    }

 private:
    void _init(
            const Shared & shared,
            const std::vector<Group> & groups) const {
        const size_t dim = shared.dim;
        shared_part_.resize(dim + 1);
        float alpha_sum = 0;
        for (size_t i = 0; i < dim; ++i) {
            float alpha = shared.alphas[i];
            alpha_sum += alpha;
            shared_part_[i] = fast_lgamma(alpha);
        }
        alpha_sum_ = alpha_sum;
        shared_part_.back() = fast_lgamma(alpha_sum);

        scores_.resize(0);
        scores_.resize(dim + 1, 0);
        for (auto const & group : groups) {
            if (group.count_sum) {
                for (size_t i = 0; i < dim; ++i) {
                    float alpha = shared.alphas[i];
                    scores_[i] += fast_lgamma(alpha + group.counts[i])
                               - shared_part_[i];
                }
                scores_.back() += shared_part_.back()
                               - fast_lgamma(alpha_sum + group.count_sum);
            }
        }
    }

    float _eval() const {
        return vector_sum(scores_.size(), scores_.data());
    }

    void _update(
            Value value,
            float old_alpha,
            float new_alpha,
            const std::vector<Group> & groups) const {
        shared_part_[value] = fast_lgamma(new_alpha);
        alpha_sum_ += static_cast<double>(new_alpha)
                    - static_cast<double>(old_alpha);
        const float alpha_sum = alpha_sum_;
        shared_part_.back() = fast_lgamma(alpha_sum);

        scores_[value] = 0;
        scores_.back() = 0;
        for (auto const & group : groups) {
            scores_[value] += fast_lgamma(new_alpha + group.counts[value])
                           - shared_part_[value];
            scores_.back() += shared_part_.back()
                           - fast_lgamma(alpha_sum + group.count_sum);
        }
    }

    mutable double alpha_sum_;
    mutable VectorFloat shared_part_;
    mutable VectorFloat scores_;
};

struct MixtureValueScorer : MixtureSlaveValueScorerMixin<Model> {
    void resize(const Shared & shared, size_t size) {
        scores_shift_.resize(size);
        scores_.resize(shared.dim);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].resize(size);
        }
    }

    void add_group(const Shared & shared, rng_t &) {
        scores_shift_.packed_add(0);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].packed_add(0);
        }
    }

    void remove_group(const Shared & shared, size_t groupid) {
        scores_shift_.packed_remove(groupid);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].packed_remove(groupid);
        }
    }

    void update_group(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t &) {
        scores_shift_[groupid] = fast_log(alpha_sum_ + group.count_sum);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value][groupid] =
                fast_log(shared.alphas[value] + group.counts[value]);
        }
    }

    void add_value(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &) {
        _update_group_value(shared, groupid, group, value);
    }

    void remove_value(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &) {
        _update_group_value(shared, groupid, group, value);
    }

    void update_all(
            const Shared & shared,
            const std::vector<Group> & groups,
            rng_t &) {
        const size_t group_count = groups.size();

        alpha_sum_ = 0;
        for (Value value = 0; value < shared.dim; ++value) {
            alpha_sum_ += shared.alphas[value];
        }
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = groups[groupid];
            for (Value value = 0; value < shared.dim; ++value) {
                scores_[value][groupid] =
                    shared.alphas[value] + group.counts[value];
            }
            scores_shift_[groupid] = alpha_sum_ + group.count_sum;
        }
        vector_log(group_count, scores_shift_.data());
        for (Value value = 0; value < shared.dim; ++value) {
            vector_log(group_count, scores_[value].data());
        }
    }

    float score_value_group(
            const Shared & shared,
            const std::vector<Group> &,
            size_t groupid,
            const Value & value,
            rng_t &) const {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        return scores_[value][groupid] - scores_shift_[groupid];
    }

    void score_value(
            const Shared & shared,
            const std::vector<Group> &,
            const Value & value,
            AlignedFloats scores_accum,
            rng_t &) const {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        vector_add_subtract(
            scores_accum.size(),
            scores_accum.data(),
            scores_[value].data(),
            scores_shift_.data());
    }

    void validate(
            const Shared & shared,
            const std::vector<Group> & groups) const {
        DIST_ASSERT_EQ(scores_.size(), (size_t)shared.dim);
        for (Value value = 0; value < shared.dim; ++value) {
            DIST_ASSERT_EQ(scores_[value].size(), groups.size());
        }
        DIST_ASSERT_EQ(scores_shift_.size(), groups.size());
    }

 private:
    void _update_group_value(
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value) {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        scores_[value][groupid] =
            fast_log(shared.alphas[value] + group.counts[value]);
        scores_shift_[groupid] = fast_log(alpha_sum_ + group.count_sum);
    }

    float alpha_sum_;
    std::vector<VectorFloat> scores_;
    VectorFloat scores_shift_;
};
};  // struct DirichletDiscrete
}   // namespace distributions
