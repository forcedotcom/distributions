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
            count_t sample_size) const
    {
        group_size -= 1;
        return -score_add_value(group_size, sample_size);
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
