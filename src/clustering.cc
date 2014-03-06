#include <algorithm>
#include <distributions/clustering.hpp>
#include <distributions/special.hpp>

namespace distributions
{

template<class count_t>
std::vector<int> Clustering<count_t>::count_assignments (
        const assignments_t & assignments)
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
// Explicit template instantiation

template<>
std::vector<int> Clustering<int>::count_assignments (const assignments_t &);

} // namespace distributions
