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

#include <unordered_map>
#include <unordered_set>
#include <distributions/common.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>

namespace distributions
{

// This is explicitly instantiated for:
// - int32_t
// To add datatypes, edit the bottom of src/clustering.cc
template<class count_t>
struct Clustering
{


//----------------------------------------------------------------------------
// Assignments

template<class Key>
struct TrivialHash
{
    typedef Key argument_type;
    typedef size_t result_type;
    size_t operator() (const Key & key) const
    {
        static_assert(sizeof(Key) <= sizeof(size_t), "invalid type");
        return key;
    }
};

typedef std::unordered_map<count_t, count_t, TrivialHash<count_t>> Assignments;

static std::vector<count_t> count_assignments (
        const Assignments & assignments);


//----------------------------------------------------------------------------
// Pitman-Yor Model

struct PitmanYor
{
    float alpha;
    float d;

    std::vector<count_t> sample_assignments (
            count_t size,
            rng_t & rng) const;

    float score_counts (
            const std::vector<count_t> & counts) const;

    float score_add_value (
            count_t group_size,
            count_t nonempty_group_count,
            count_t sample_size,
            count_t empty_group_count = 1) const
    {
        // What is the probability (score) of adding a customer
        // to a table which currently has:
        //
        // group_size people sitting at it (can be zero)
        // nonempty_group_count tables that have people sitting at them
        // sample_size people seated total
        //
        // In particular, if group_size == 0, this is the prob of sitting
        // at a new table. In that case, nonempty_group_count does not
        // include this "new" table, as it is obviously unoccupied.

        if (group_size == 0) {
            float numer = alpha + d * nonempty_group_count;
            float denom = (sample_size + alpha) * empty_group_count;
            return fast_log(numer / denom);
        } else {
            return fast_log((group_size - d) / (sample_size + alpha));
        }
    }

    float score_remove_value(
            count_t group_size,
            count_t nonempty_group_count,
            count_t sample_size,
            count_t empty_group_count = 1) const
    {
        group_size -= 1;
        if (group_size == 0) {
            --nonempty_group_count;
        }
        sample_size -= 1;

        return -score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count);
    }

    // HACK gcc doesn't want Mixture defined outside of PitmanYor
    struct Mixture
    {
        typedef PitmanYor Model;

        std::vector<count_t> counts;
        std::unordered_set<size_t, TrivialHash<size_t>> empty_groupids;
        count_t sample_size;
        VectorFloat shifted_scores;

        void init (const Model & model)
        {
            const size_t group_count = counts.size();
            sample_size = 0;
            shifted_scores.resize(group_count);
            empty_groupids.clear();
            for (size_t i = 0; i < group_count; ++i) {
                sample_size += counts[i];
                if (counts[i]) {
                    _update_nonempty_group(model, i);
                } else {
                    empty_groupids.insert(i);
                }
            }
            DIST_ASSERT(empty_groupids.size(), "missing empty groups");
            _update_empty_groups(model);
            _validate();
        }

        bool add_value (
                const Model & model,
                size_t groupid,
                count_t count = 1)
        {
            DIST_ASSERT1(count, "cannot add zero values");
            DIST_ASSERT2(groupid < counts.size(), "bad groupid: " << groupid);

            const bool add_group = (counts[groupid] == 0);
            counts[groupid] += count;
            sample_size += count;
            _update_nonempty_group(model, groupid);

            if (DIST_UNLIKELY(add_group)) {
                empty_groupids.erase(groupid);
                empty_groupids.insert(counts.size());
                counts.push_back(0);
                shifted_scores.push_back(0);
                _update_empty_groups(model);
            }
            _validate();

            return add_group;
        }

        bool remove_value (
                const Model & model,
                size_t groupid,
                count_t count = 1)
        {
            DIST_ASSERT1(count, "cannot remove zero values");
            DIST_ASSERT2(groupid < counts.size(), "bad groupid: " << groupid);
            DIST_ASSERT2(
                counts[groupid],
                "cannot remove value from empty group");
            DIST_ASSERT2(
                count <= counts[groupid],
                "cannot remove more values than are in group");

            counts[groupid] -= count;
            sample_size -= count;
            const bool remove_group = (counts[groupid] == 0);

            if (DIST_LIKELY(not remove_group)) {
                _update_nonempty_group(model, groupid);
            } else {
                const size_t group_count = counts.size() - 1;
                if (groupid != group_count) {
                    if (counts.back() == 0) {
                        empty_groupids.erase(group_count);
                        empty_groupids.insert(groupid);
                    } else {
                        counts[groupid] = counts.back();
                        shifted_scores[groupid] = shifted_scores.back();
                    }
                }
                counts.pop_back();
                shifted_scores.pop_back();;
                _update_empty_groups(model);
            }
            _validate();

            return remove_group;
        }

        void score (const Model & model, VectorFloat & scores) const
        {
            const size_t size = counts.size();
            const float shift = -fast_log(sample_size + model.alpha);
            const float * __restrict__ in = VectorFloat_data(shifted_scores);
            float * __restrict__ out = VectorFloat_data(scores);

            for (size_t i = 0; i < size; ++i) {
                out[i] = in[i] + shift;
            }
        }

    private:

        void _validate () const
        {
            DIST_ASSERT1(empty_groupids.size(), "missing empty groups");
            if (DIST_DEBUG_LEVEL >= 2) {
                for (size_t i = 0; i < counts.size(); ++i) {
                    bool count_is_zero = (counts[i] == 0);
                    bool is_empty =
                        (empty_groupids.find(i) != empty_groupids.end());
                    DIST_ASSERT_EQ(count_is_zero, is_empty);
                }
            }
        }

        void _update_nonempty_group (const Model & model, size_t groupid)
        {
            const auto group_size = counts[groupid];
            DIST_ASSERT2(group_size, "expected nonempty group");
            shifted_scores[groupid] = fast_log(group_size - model.d);
        }

        void _update_empty_groups (const Model & model)
        {
            size_t empty_group_count = empty_groupids.size();
            size_t nonempty_group_count = counts.size() - empty_group_count;
            float numer = model.alpha + model.d * nonempty_group_count;
            float denom = empty_group_count;
            const float shifted_score = fast_log(numer / denom);
            for (size_t i : empty_groupids) {
                shifted_scores[i] = shifted_score;
            }
        }
    };
};


//----------------------------------------------------------------------------
// Low-Entropy Model

struct LowEntropy
{
    count_t dataset_size;

    std::vector<count_t> sample_assignments (
            count_t sample_size,
            rng_t & rng) const;

    float score_counts (const std::vector<count_t> & counts) const;

    float score_add_value (
            count_t group_size,
            count_t nonempty_group_count,
            count_t sample_size,
            count_t empty_group_count = 1) const
    {
        if (DIST_DEBUG_LEVEL >= 1) {
            DIST_ASSERT_LT(sample_size, dataset_size);
            DIST_ASSERT_LT(0, empty_group_count);
        }

        if (group_size == 0) {
            float score = -fast_log(empty_group_count);
            if (sample_size + 1 < dataset_size) {
                score += _approximate_postpred_correction(sample_size + 1);
            }
            return score;
        }

        // see `python derivations/clustering.py fastlog`
        const count_t very_large = 10000;
        float bigger = 1.f + group_size;
        if (group_size > very_large) {
            return 1.f + fast_log(bigger);
        } else {
            return fast_log(bigger / group_size) * group_size
                 + fast_log(bigger);
        }
    }

    float score_remove_value (
            count_t group_size,
            count_t nonempty_group_count,
            count_t sample_size,
            count_t empty_group_count = 1) const
    {
        if (DIST_DEBUG_LEVEL >= 1) {
            DIST_ASSERT_LT(0, sample_size);
        }

        group_size -= 1;
        return -score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count);
    }

    float log_partition_function (count_t sample_size) const;

private:

    // ad hoc approximation,
    // see `python derivations/clustering.py postpred`
    // see `python derivations/clustering.py approximations`
    float _approximate_postpred_correction (float sample_size) const
    {
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_LT(0, sample_size);
            DIST_ASSERT_LT(sample_size, dataset_size);
        }

        float exponent = 0.45f - 0.1f / sample_size - 0.1f / dataset_size;
        float scale = dataset_size / sample_size;
        return fast_log(scale) * exponent;
    }

    float _approximate_dataprob_correction(count_t sample_size) const;
};

}; // struct Clustering<count_t>
} // namespace distributions
