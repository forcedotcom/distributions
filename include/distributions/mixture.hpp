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
#include <unordered_set>
#include <unordered_map>
#include <type_traits>
#include <distributions/common.hpp>
#include <distributions/vector.hpp>
#include <distributions/trivial_hash.hpp>
#include <distributions/random_fwd.hpp>

namespace distributions {

// --------------------------------------------------------------------------
// Mixture Driver
//
// This interface maintains contiguous groupids for vectorized scoring
// while maintaining a fixed number of empty groups.
// Specific models may use this class, or maintain custom cached scores.

template<class Model_, class count_t>
struct MixtureDriver {
    typedef Model_ Model;
    typedef std::unordered_set<size_t, TrivialHash<size_t>> IdSet;

    std::vector<count_t> & counts() { return counts_; }
    const std::vector<count_t> & counts() const { return counts_; }
    count_t counts(size_t groupid) const { return counts_[groupid]; }
    const IdSet & empty_groupids() const { return empty_groupids_; }
    size_t sample_size() const { return sample_size_; }

    void init(const Model &) {
        empty_groupids_.clear();
        sample_size_ = 0;

        const size_t group_count = counts_.size();
        for (size_t i = 0; i < group_count; ++i) {
            sample_size_ += counts_[i];
            if (counts_[i] == 0) {
                empty_groupids_.insert(i);
            }
        }
        _validate();
    }

    bool add_value(
            const Model &,
            size_t groupid,
            count_t count = 1) {
        DIST_ASSERT1(count, "cannot add zero values");
        DIST_ASSERT2(groupid < counts_.size(), "bad groupid: " << groupid);

        const bool add_group = (counts_[groupid] == 0);
        counts_[groupid] += count;
        sample_size_ += count;

        if (DIST_UNLIKELY(add_group)) {
            empty_groupids_.erase(groupid);
            empty_groupids_.insert(counts_.size());
            counts_.push_back(0);
            _validate();
        }

        return add_group;
    }

    bool remove_value(
            const Model &,
            size_t groupid,
            count_t count = 1) {
        DIST_ASSERT1(count, "cannot remove zero values");
        DIST_ASSERT2(groupid < counts_.size(), "bad groupid: " << groupid);
        DIST_ASSERT2(counts_[groupid], "cannot remove value from empty group");
        DIST_ASSERT2(count <= counts_[groupid],
            "cannot remove more values than are in group");

        counts_[groupid] -= count;
        sample_size_ -= count;
        const bool remove_group = (counts_[groupid] == 0);

        if (DIST_UNLIKELY(remove_group)) {
            const size_t group_count = counts_.size() - 1;
            if (groupid != group_count) {
                counts_[groupid] = counts_.back();
                if (counts_.back() == 0) {
                    empty_groupids_.erase(group_count);
                    empty_groupids_.insert(groupid);
                }
            }
            counts_.pop_back();
            _validate();
        }

        return remove_group;
    }

    void score_value(const Model & model, AlignedFloats scores) const {
        DIST_THIS_SLOW_FALLBACK_SHOULD_BE_OVERRIDDEN

        if (DIST_DEBUG_LEVEL >= 1) {
            DIST_ASSERT_EQ(scores.size(), counts_.size());
        }

        const count_t group_count = counts_.size();
        const count_t empty_group_count = empty_groupids_.size();
        const count_t nonempty_group_count = group_count - empty_group_count;
        for (size_t i = 0; i < group_count; ++i) {
            scores[i] = model.score_add_value(
                counts_[i],
                nonempty_group_count,
                sample_size_,
                empty_group_count);
        }
    }

    float score_data(const Model & model) const {
        return model.score_counts(counts_);
    }

 private:
    std::vector<count_t> counts_;
    IdSet empty_groupids_;
    count_t sample_size_;

    void _validate() const {
        DIST_ASSERT1(empty_groupids_.size(), "missing empty groups");
        if (DIST_DEBUG_LEVEL >= 2) {
            for (size_t i = 0; i < counts_.size(); ++i) {
                bool count_is_zero = (counts_[i] == 0);
                bool is_empty =
                    (empty_groupids_.find(i) != empty_groupids_.end());
                DIST_ASSERT_EQ(count_is_zero, is_empty);
            }
        }
    }
};


// --------------------------------------------------------------------------
// Mixture Slave

template<class Model>
struct MixtureSlaveGroups {
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;
    typedef typename Model::Value Value;

    std::vector<Group> & groups() { return groups_; }
    Group & groups(size_t groupid) {
        DIST_ASSERT1(groupid < groups_.size(), "bad groupid: " << groupid);
        return groups_[groupid];
    }

    const std::vector<Group> & groups() const { return groups_; }
    const Group & groups(size_t groupid) const {
        DIST_ASSERT1(groupid < groups_.size(), "bad groupid: " << groupid);
        return groups_[groupid];
    }

    // add_group is called whenever driver.add_value returns true
    void add_group(
            const Shared & shared,
            rng_t & rng) {
        groups_.packed_add().init(shared, rng);
    }

    // remove_group is called whenever driver.remove_value returns true
    void remove_group(
            const Shared &,
            size_t groupid) {
        groups_.packed_remove(groupid);
    }

    void add_value(
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng) {
        groups(groupid).add_value(shared, value, rng);
    }

    void remove_value(
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng) {
        groups(groupid).remove_value(shared, value, rng);
    }

    void validate(const Shared & shared) const {
        for (auto const & group : groups_) {
            group.validate(shared);
        }
    }

 private:
    Packed_<Group> groups_;
};

template<class Model_, class Derived>
struct MixtureSlaveDataScorerMixin {
    const Derived & self() const {
        return static_cast<const Derived &>(*this);
    }

    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;

    void score_data_grid(
            const std::vector<Shared> & shareds,
            const std::vector<Group> & groups,
            AlignedFloats scores_out,
            rng_t & rng) const {
        DIST_ASSERT_EQ(shareds.size(), scores_out.size());
        for (size_t i = 0, size = scores_out.size(); i < size; ++i) {
            scores_out[i] = self().score_data(shareds[i], groups, rng);
        }
    }

    void validate(const Shared &, const std::vector<Group> &) const {}
};

template<class Model>
struct SmallMixtureSlaveDataScorer :  // NOLINT(*)
    MixtureSlaveDataScorerMixin<Model, SmallMixtureSlaveDataScorer<Model>> {
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;

    float score_data(
            const Shared & shared,
            const std::vector<Group> & groups,
            rng_t & rng) const {
        float score = 0;
        for (const Group & group : groups) {
            score += group.score_data(shared, rng);
        }
        return score;
    }
};

template<class Model_>
struct MixtureSlaveValueScorerMixin {
    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;

    void resize(const Shared &, size_t) {}
    void add_group(const Shared &, rng_t &) {}
    void remove_group(const Shared &, size_t) {}
    void update_group(const Shared &, size_t, const Group &, rng_t &) {}
    void update_all(const Shared &, const std::vector<Group> &, rng_t &) {}

    void add_value(
            const Shared &,
            size_t,
            const Group &,
            const Value &,
            rng_t &) {}

    void remove_value(
            const Shared &,
            size_t,
            const Group &,
            const Value &,
            rng_t &) {}

    void validate(const Shared &, const std::vector<Group> &) const {}
};

template<class Model>
struct SmallMixtureSlaveValueScorer : MixtureSlaveValueScorerMixin<Model> {
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;

    float score_value_group(
            const Shared & shared,
            const std::vector<Group> & groups,
            size_t groupid,
            const Value & value,
            rng_t & rng) const {
        DIST_THIS_SLOW_FALLBACK_SHOULD_BE_OVERRIDDEN

        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_LT(groupid, groups.size());
        }

        return groups[groupid].score_value(shared, value, rng);
    }

    void score_value(
            const Shared & shared,
            const std::vector<Group> & groups,
            const Value & value,
            AlignedFloats scores_accum,
            rng_t & rng) const {
        DIST_THIS_SLOW_FALLBACK_SHOULD_BE_OVERRIDDEN

        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_EQ(scores_accum.size(), groups.size());
        }

        const size_t group_count = groups.size();
        for (size_t i = 0; i < group_count; ++i) {
            scores_accum[i] += groups[i].score_value(shared, value, rng);
        }
    }
};

template<
    class Model,  // NOLINT(*)
    class DataScorer = SmallMixtureSlaveDataScorer<Model>,
    class ValueScorer = SmallMixtureSlaveValueScorer<Model>>
struct MixtureSlave {
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;

    std::vector<Group> & groups() { return groups_.groups(); }
    Group & groups(size_t i) { return groups_.groups(i); }
    const std::vector<Group> & groups() const { return groups_.groups(); }
    const Group & groups(size_t i) const { return groups_.groups(i); }

    void init(
            const Shared & shared,
            rng_t & rng) {
        value_scorer_.resize(shared, groups().size());
        value_scorer_.update_all(shared, groups(), rng);
    }

    void add_group(
            const Shared & shared,
            rng_t & rng) {
        const size_t groupid = groups().size();
        groups_.add_group(shared, rng);
        value_scorer_.add_group(shared, rng);
        value_scorer_.update_group(shared, groupid, groups(groupid), rng);
    }

    void remove_group(
            const Shared & shared,
            size_t groupid) {
        groups_.remove_group(shared, groupid);
        value_scorer_.remove_group(shared, groupid);
    }

    void add_value(
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng) {
        groups_.add_value(shared, groupid, value, rng);
        value_scorer_.add_value(shared, groupid, groups(groupid), value, rng);
    }

    void remove_value(
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng) {
        groups_.remove_value(shared, groupid, value, rng);
        value_scorer_.remove_value(
            shared,
            groupid,
            groups(groupid),
            value,
            rng);
    }

    float score_value_group(
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng) const {
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_LT(groupid, groups().size());
        }
        return value_scorer_.score_value_group(
            shared,
            groups(),
            groupid,
            value,
            rng);
    }

    void score_value(
            const Shared & shared,
            const Value & value,
            AlignedFloats scores_accum,
            rng_t & rng) const {
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_EQ(scores_accum.size(), groups().size());
        }
        value_scorer_.score_value(shared, groups(), value, scores_accum, rng);
    }

    float score_data(
            const Shared & shared,
            rng_t & rng) const {
        return data_scorer_.score_data(shared, groups(), rng);
    }

    void score_data_grid(
            const std::vector<Shared> & shareds,
            AlignedFloats scores_out,
            rng_t & rng) const {
        data_scorer_.score_data_grid(shareds, groups(), scores_out, rng);
    }

    void validate(const Shared & shared) const {
        groups_.validate(shared);
        value_scorer_.validate(shared, groups());
        data_scorer_.validate(shared, groups());
    }

 private:
    MixtureSlaveGroups<Shared> groups_;
    ValueScorer value_scorer_;
    DataScorer data_scorer_;
};


// --------------------------------------------------------------------------
// Mixture Id Tracker
//
// This interface tracks a mapping between contiguous "packed" group ids
// and fixed unique "global" ids.  Packed ids can change when groups are
// added or removed, but global ids never change.

struct MixtureIdTracker {
    typedef uint32_t Id;

    void init(size_t group_count = 0) {
        packed_to_global_.clear();
        global_to_packed_.clear();
        global_size_ = 0;
        for (size_t i = 0; i < group_count; ++i) {
            add_group();
        }
    }

    void add_group() {
        const Id packed = packed_to_global_.size();
        const Id global = global_size_++;
        packed_to_global_.packed_add(global);
        global_to_packed_.insert(std::make_pair(global, packed));
    }

    void remove_group(Id packed) {
        DIST_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        const Id global = packed_to_global_[packed];
        DIST_ASSERT1(global < global_size(), "bad global id: " << global);
        global_to_packed_.erase(global);
        packed_to_global_.packed_remove(packed);
        if (packed != packed_size()) {
            const Id global = packed_to_global_[packed];
            DIST_ASSERT1(global < global_size(), "bad global id: " << global);
            auto i = global_to_packed_.find(global);
            DIST_ASSERT1(
                i != global_to_packed_.end(),
                "stale global id: " << global);
            i->second = packed;
        }
    }

    Id packed_to_global(Id packed) const {
        DIST_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        Id global = packed_to_global_[packed];
        DIST_ASSERT1(global < global_size(), "bad global id: " << global);
        return global;
    }

    Id global_to_packed(Id global) const {
        DIST_ASSERT1(global < global_size(), "bad global id: " << global);
        auto i = global_to_packed_.find(global);
        DIST_ASSERT1(
            i != global_to_packed_.end(),
            "stale global id: " << global);
        Id packed = i->second;
        DIST_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        return packed;
    }

    size_t packed_size() const { return packed_to_global_.size(); }
    size_t global_size() const { return global_size_; }

 private:
    Packed_<Id> packed_to_global_;
    std::unordered_map<Id, Id, TrivialHash<Id>> global_to_packed_;
    size_t global_size_;
};

}   // namespace distributions
