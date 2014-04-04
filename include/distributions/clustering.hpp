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
#include <algorithm>
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

    //------------------------------------------------------------------------
    // Mixture

    struct Mixture
    {
        std::vector<count_t> counts;
        size_t empty_groupid;
        count_t sample_size;
        VectorFloat shifted_scores;
    };

    private:

    void _mixture_validate (Mixture & mixture) const
    {
        if (DIST_DEBUG_LEVEL >= 3) {
            size_t empty_group_count = std::count_if(
                mixture.counts.begin(),
                mixture.counts.end(),
                [&](count_t count){ return count == 0; });
            DIST_ASSERT(
                empty_group_count == 1,
                "expected 1 empty group, actual " << empty_group_count);
        }
    }

    void _mixture_update_group (Mixture & mixture, size_t groupid) const
    {
        const auto nonempty_group_count = mixture.counts.size() - 1;
        const auto group_size = mixture.counts[groupid];
        mixture.shifted_scores[groupid] =
            fast_log(group_size ? group_size - d
                                : alpha + d * nonempty_group_count);
    }

    public:

    void mixture_init (Mixture & mixture) const
    {
        const size_t group_count = mixture.counts.size();
        mixture.sample_size = 0;
        mixture.shifted_scores.resize(group_count);
        for (size_t i = 0; i < group_count; ++i) {
            mixture.sample_size += mixture.counts[i];
            _mixture_update_group(mixture, i);
            if (mixture.counts[i] == 0) {
                mixture.empty_groupid = i;
            }
        }
        _mixture_validate(mixture);
    }

    bool mixture_add_value (Mixture & mixture, size_t groupid) const
    {
        mixture.counts[groupid] += 1;
        mixture.sample_size += 1;
        _mixture_update_group(mixture, groupid);

        bool add_group = (groupid == mixture.empty_groupid);
        if (DIST_UNLIKELY(add_group)) {
            mixture.empty_groupid = mixture.counts.size();
            mixture.counts.push_back(0);
            mixture.shifted_scores.push_back(0);
            _mixture_update_group(mixture, mixture.empty_groupid);
        }
        _mixture_validate(mixture);

        return add_group;
    }

    bool mixture_remove_value (Mixture & mixture, size_t groupid) const
    {
        DIST_ASSERT2(
            groupid != mixture.empty_groupid,
            "cannot remove value from empty group");

        mixture.counts[groupid] -= 1;
        mixture.sample_size -= 1;

        bool remove_group = (mixture.counts[groupid] == 0);
        if (DIST_LIKELY(not remove_group)) {
            _mixture_update_group(mixture, groupid);
        } else {
            const size_t group_count = mixture.counts.size() - 1;
            if (groupid != group_count) {
                mixture.counts[groupid] = mixture.counts.back();
                mixture.shifted_scores[groupid] = mixture.shifted_scores.back();
                if (mixture.empty_groupid == group_count) {
                    mixture.empty_groupid = groupid;
                }
            }
            mixture.counts.resize(group_count);
            mixture.shifted_scores.resize(group_count);
            _mixture_update_group(mixture, mixture.empty_groupid);
        }
        _mixture_validate(mixture);

        return remove_group;
    }

    void mixture_score (const Mixture & mixture, VectorFloat & scores) const
    {
        const size_t size = mixture.counts.size();
        const float shift = -fast_log(mixture.sample_size + alpha);
        const float * __restrict__ in =
            VectorFloat_data(mixture.shifted_scores);
        float * __restrict__ out = VectorFloat_data(scores);

        for (size_t i = 0; i < size; ++i) {
            out[i] = in[i] + shift;
        }
    }
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
