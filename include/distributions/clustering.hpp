#pragma once

#include <unordered_map>
#include <distributions/common.hpp>
#include <distributions/random.hpp>

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

typedef std::unordered_map<count_t, count_t, trivial_hash> assignments_t;

static std::vector<int> count_assignments (const assignments_t & assignments);


//----------------------------------------------------------------------------
// Models

struct PitmanYor
{
    float alpha;
    float d;

    std::vector<count_t> sample_assignments (
            size_t size,
            rng_t & rng) const;

    void score_counts (
            const std::vector<count_t> & counts) const;

    float score_add_value (
            count_t this_group_size,
            count_t total_group_count,
            count_t total_value_count) const
    {
        // What is the probability (score) of adding a customer
        // to a table which currently has:
        //
        // this_group_size people sitting at it (can be zero)
        // total_group_count tables that have people sitting at them
        // total_value_count people seated total
        //
        // In particular, if this_group_size == 0, this is the prob of sitting
        // at a new table. In that case, total_group_count does not
        // include this "new" table, as it is obviously unoccupied.

        if (this_group_size == 0) {
            return fast_log(
                (alpha + d * total_group_count) / (total_value_count + alpha));
        } else {
            return fast_log(
                (this_group_size - d) / (total_value_count + alpha));
        }
    }

    float score_remove_value(
            count_t this_group_size,
            count_t total_group_count,
            count_t total_value_count) const
    {
        this_group_size -= 1;
        if (this_group_size == 0) {
            --total_group_count;
        }
        total_value_count -= 1;

        return -score_add_value(
            this_group_size,
            total_group_count,
            total_value_count);
    }
};

}; // struct Clustering<count_t>
} // namespace distributions
