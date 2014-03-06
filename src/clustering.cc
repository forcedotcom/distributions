#include <algorithm>
#include <distributions/clustering.hpp>
#include <distributions/special.hpp>

namespace distributions
{

//----------------------------------------------------------------------------
// Assignments

template<class count_t>
std::vector<count_t> Clustering<count_t>::count_assignments (
        const Assignments & assignments)
{
    // Count group sizes in an assignment vector with the following properties:
    // 0 is the first group
    // there are no empty groups
    // the group IDs are contiguous.

    std::vector<count_t> counts;
    for (auto pair : assignments) {
        count_t gid = pair.second;
        if (gid >= counts.size()) {
            counts.resize(gid + 1, 0);
        }
        counts[gid]++;
    }

#ifndef NDEBUG2
    if (not counts.empty()) {
        count_t min_count = * std::min_element(counts.begin(), counts.end());
        DIST_ASSERT(min_count > 0, "groups are not contiguous");
    }
#endif // NDEBUG2

    return counts;
}


//----------------------------------------------------------------------------
// Models

template<class count_t>
std::vector<count_t> Clustering<count_t>::PitmanYor::sample_assignments (
        count_t size,
        rng_t & rng) const
{
    // Note that we can ignore the constant shift of -log(size + alpha) in
    //
    //   float py.score_add_value(
    //          count_t this_group_size,
    //          count_t total_group_count,
    //          count_t size)
    //   {
    //     return this_group_size == 0
    //          ? fast_log( (alpha + d * total_group_count)
    //                     / (size + alpha) )
    //          : fast_log((this_group_size - d) / (size + alpha));
    //   }
    //
    // which then permits caching of each table's score.

    DIST_ASSERT(float(size) + 1.f > float(size), "underflow expected");

    std::vector<count_t> assignments(size);
    std::vector<float> likelihoods;
    likelihoods.reserve(100);  // just pick something safe


    // initialize empty table
    count_t table_count = 0;
    const float py_likelihood_new = 1 - d;
    const float py_likelihood_empty = alpha;
    likelihoods.push_back(py_likelihood_empty);


    // add first entry
    {
        count_t i = 0;
        count_t assign = 0;
        assignments[i] = assign;

        table_count = 1;
        const float py_likelihood_empty = alpha + d * table_count;
        likelihoods.push_back(py_likelihood_empty);
        likelihoods[assign] = py_likelihood_new;
    }


    // add all remaining entries
    for (count_t i = 1; i < size; ++i) {

        // This is cool - for fixed alpha, d, the likelihood will roughly
        // exponentially decay along the likelihood vector.  And in sampling
        // we linearly scan from the front, so we only need to examine an
        // expected constant number of entries.  This results in a expected
        // runtime of the whole sampler linear in size.
        float total = i + alpha;
        count_t assign = sample_from_likelihoods(rng, likelihoods, total);
        assignments[i] = assign;

        if (assign == table_count) {

            // new table
            table_count += 1;
            const float py_likelihood_empty = alpha + d * table_count;
            likelihoods.push_back(py_likelihood_empty);
            likelihoods[assign] = py_likelihood_new;

        } else {

            // existing table
            likelihoods[assign] += 1.0f;
        }
    }

    return assignments;
}

template<class count_t>
float Clustering<count_t>::PitmanYor::score_counts (
        const std::vector<count_t> & counts) const
{
    double score = 0.0;
    count_t totalfpos = 0;

    for (count_t c = 0; c < counts.size(); ++c) {
        for (count_t fpos = 0; fpos < counts[c]; ++fpos) {
            if (fpos == 0) {
                score += score_add_value(fpos, c, totalfpos);
            } else {
                score += score_add_value(fpos, c + 1, totalfpos);
            }
            ++totalfpos;
        }
    }

    return score;
}


//----------------------------------------------------------------------------
// Explicit template instantiation

template std::vector<int> Clustering<int>::count_assignments (
        const Assignments &);
template struct Clustering<int>::PitmanYor;

} // namespace distributions
