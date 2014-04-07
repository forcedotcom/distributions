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
#include <distributions/common.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>

namespace distributions
{

// This is explicitly instantiated for: int
template<class count_t>
struct Clustering
{


//----------------------------------------------------------------------------
// Assignments

struct trivial_hash
{
    typedef count_t argument_type;
    typedef size_t result_type;
    size_t operator() (const count_t & key) const
    {
        static_assert(sizeof(count_t) <= sizeof(size_t), "invalid count_t");
        return key;
    }
};

typedef std::unordered_map<count_t, count_t, trivial_hash> Assignments;

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
            count_t group_count,
            count_t sample_size) const
    {
        // What is the probability (score) of adding a customer
        // to a table which currently has:
        //
        // group_size people sitting at it (can be zero)
        // group_count tables that have people sitting at them
        // sample_size people seated total
        //
        // In particular, if group_size == 0, this is the prob of sitting
        // at a new table. In that case, group_count does not
        // include this "new" table, as it is obviously unoccupied.

        if (group_size == 0) {
            return fast_log(
                (alpha + d * group_count) / (sample_size + alpha));
        } else {
            return fast_log(
                (group_size - d) / (sample_size + alpha));
        }
    }

    float score_remove_value(
            count_t group_size,
            count_t group_count,
            count_t sample_size) const
    {
        group_size -= 1;
        if (group_size == 0) {
            --group_count;
        }
        sample_size -= 1;

        return -score_add_value(group_size, group_count, sample_size);
    }

    struct Mixture
    {
        typedef PitmanYor Model;

        std::vector<count_t> counts;
        size_t empty_groupid;
        count_t sample_size;
        VectorFloat shifted_scores;

    private:

        void _validate () const
        {
            DIST_ASSERT2(
                counts[empty_groupid] == 0,
                "empty_group is not empty");
            if (DIST_DEBUG_LEVEL >= 3) {
                for (size_t i = 0; i < counts.size(); ++i) {
                    if (i != empty_groupid) {
                        DIST_ASSERT(counts[i], "extra empty group: " << i);
                    }
                }
            }
        }

        void _update_group (const Model & model, size_t groupid)
        {
            const auto nonempty_group_count = counts.size() - 1;
            const auto group_size = counts[groupid];
            shifted_scores[groupid] =
                fast_log(group_size
                        ? group_size - model.d
                        : model.alpha + model.d * nonempty_group_count);
        }

    public:

        void init (const Model & model)
        {
            const size_t group_count = counts.size();
            sample_size = 0;
            shifted_scores.resize(group_count);
            for (size_t i = 0; i < group_count; ++i) {
                sample_size += counts[i];
                _update_group(model, i);
                if (counts[i] == 0) {
                    empty_groupid = i;
                }
            }
            _validate();
        }

        bool add_value (
                const Model & model,
                size_t groupid,
                count_t count = 1)
        {
            counts[groupid] += count;
            sample_size += count;
            _update_group(model, groupid);

            bool add_group = (groupid == empty_groupid);
            if (DIST_UNLIKELY(add_group)) {
                empty_groupid = counts.size();
                counts.push_back(0);
                shifted_scores.push_back(0);
                _update_group(model, empty_groupid);
            }
            _validate();

            return add_group;
        }

        bool remove_value (
                const Model & model,
                size_t groupid,
                count_t count = 1)
        {
            DIST_ASSERT2(
                groupid != empty_groupid,
                "cannot remove value from empty group");
            DIST_ASSERT2(
                count <= counts[groupid],
                "cannot remove more values than are in group");

            counts[groupid] -= count;
            sample_size -= count;

            bool remove_group = (counts[groupid] == 0);
            if (DIST_LIKELY(not remove_group)) {
                _update_group(model, groupid);
            } else {
                const size_t group_count = counts.size() - 1;
                if (groupid != group_count) {
                    counts[groupid] = counts.back();
                    shifted_scores[groupid] = shifted_scores.back();
                    if (empty_groupid == group_count) {
                        empty_groupid = groupid;
                    }
                }
                counts.resize(group_count);
                shifted_scores.resize(group_count);
                _update_group(model, empty_groupid);
            }
            _validate();

            return remove_group;
        }

        void score (const Model & model, VectorFloat & scores) const
        {
            const size_t size = counts.size();
            const float shift = -fast_log(sample_size + model.alpha);
            const float * __restrict__ in =
                VectorFloat_data(shifted_scores);
            float * __restrict__ out = VectorFloat_data(scores);

            for (size_t i = 0; i < size; ++i) {
                out[i] = in[i] + shift;
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
            count_t group_count,
            count_t sample_size) const
    {
        // see `python derivations/clustering.py fast_log`
        const count_t very_large = 10000;

        if (group_size == 0) {
            if (sample_size == dataset_size) {
                return 0.f;
            } else {
                return approximate_postpred_correction(sample_size);
            }
        } else if (group_size > very_large) {
            float bigger = 1.f + group_size;
            return 1.f + fast_log(bigger);
        } else {
            float bigger = 1.f + group_size;
            return fast_log(bigger / group_size) * group_size
                 + fast_log(bigger);
        }
    }

    float score_remove_value (
            count_t group_size,
            count_t group_count,
            count_t sample_size) const
    {
        group_size -= 1;
        return -score_add_value(group_size, group_count, sample_size);
    }

private:

    // ad hoc approximation,
    // see `python derivations/clustering.py postpred`
    // see `python derivations/clustering.py approximations`
    float approximate_postpred_correction (float sample_size) const
    {
        float exponent = 0.45f - 0.1f / sample_size - 0.1f / dataset_size;
        float scale = dataset_size / sample_size;
        return fast_log(scale) * exponent;
    }
};

}; // struct Clustering<count_t>
} // namespace distributions
