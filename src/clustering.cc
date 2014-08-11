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

#include <algorithm>
#include <distributions/clustering.hpp>
#include <distributions/special.hpp>

namespace distributions {

// --------------------------------------------------------------------------
// Assignments

template<class count_t>
std::vector<count_t> Clustering<count_t>::count_assignments(
        const Assignments & assignments) {
    // Count group sizes in an assignment vector with the following properties:
    // 0 is the first group
    // there are no empty groups
    // the group IDs are contiguous.

    std::vector<count_t> counts;
    for (auto pair : assignments) {
        size_t gid = pair.second;
        if (DIST_UNLIKELY(gid >= counts.size())) {
            counts.resize(gid + 1, 0);
        }
        ++counts[gid];
    }

    if (DIST_DEBUG_LEVEL >= 2) {
        if (not counts.empty()) {
            count_t min_count =
                * std::min_element(counts.begin(), counts.end());
            DIST_ASSERT(min_count > 0, "groups are not contiguous");
        }
    }

    return counts;
}


// --------------------------------------------------------------------------
// Pitman-Yor Model

template<class count_t>
std::vector<count_t> Clustering<count_t>::PitmanYor::sample_assignments(
        count_t size,
        rng_t & rng) const {
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

    DIST_ASSERT(
        static_cast<float>(size) + 1.f > static_cast<float>(size),
        "underflow expected");

    std::vector<count_t> assignments(size);
    std::vector<float> likelihoods;
    likelihoods.reserve(100);  // just pick something safe


    // initialize empty table
    count_t table_count = 0;
    const float py_likelihood_new = 1 - d;
    const float py_likelihood_empty = alpha;
    likelihoods.push_back(py_likelihood_empty);


    // add first entry
    if (DIST_LIKELY(size)) {
        count_t i = 0;
        count_t assign = 0;
        assignments[i] = assign;

        table_count = 1;
        const float py_likelihood_empty = alpha + d * table_count;
        likelihoods.push_back(py_likelihood_empty);
        likelihoods[assign] = py_likelihood_new;
    }


    // add all remaining entries
    for (count_t i = 1; DIST_LIKELY(i < size); ++i) {
        // This is cool - for fixed alpha, d, the likelihood will roughly
        // exponentially decay along the likelihood vector.  And in sampling
        // we linearly scan from the front, so we only need to examine an
        // expected constant number of entries.  This results in a expected
        // runtime of the whole sampler linear in size.
        float total = i + alpha;
        count_t assign = sample_from_likelihoods(rng, likelihoods, total);
        assignments[i] = assign;

        if (DIST_UNLIKELY(assign == table_count)) {
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

inline float fast_log_ratio(float numer, float denom) {
    return fast_log(numer / denom);
}

inline float fast_lgamma_ratio(float start, size_t count) {
    return fast_lgamma(start + count) - fast_lgamma(start);
}

template<class count_t>
float Clustering<count_t>::PitmanYor::score_counts(
        const std::vector<count_t> & counts) const {
    double score = 0.0;
    size_t sample_size = 0;
    size_t nonempty_group_count = 0;

    for (size_t count : counts) {
        if (count) {
            if (count == 1) {
                score += fast_log_ratio(
                    alpha + d * nonempty_group_count,
                    alpha + sample_size);

            } else if (count == 2) {
                score += fast_log_ratio(
                    (alpha + d * nonempty_group_count) * (1 - d),
                    (alpha + sample_size) * (alpha + sample_size + 1));

            } else {
                score += fast_log(alpha + d * nonempty_group_count);
                score += fast_lgamma_ratio(1 - d, count - 1);
                score -= fast_lgamma_ratio(alpha + sample_size, count);
            }

            nonempty_group_count += 1;
            sample_size += count;
        }
    }

    return score;
}

// --------------------------------------------------------------------------
// Low-Entropy Model

// this code was generated by derivations/clustering.py
static const float log_partition_function_table[48] = {
    0.00000000, 0.00000000, 1.60943791, 3.68887945, 6.07993320,
    8.70549682, 11.51947398, 14.49108422, 17.59827611, 20.82445752,
    24.15668300, 27.58456586, 31.09958507, 34.69462231, 38.36364086,
    42.10145572, 45.90356476, 49.76602176, 53.68533918, 57.65841234,
    61.68245958, 65.75497413, 69.87368527, 74.03652635, 78.24160846,
    82.48719834, 86.77169993, 91.09363859, 95.45164780, 99.84445762,
    104.27088480, 108.72982416, 113.22024112, 117.74116515, 122.29168392,
    126.87093829, 131.47811772, 136.11245629, 140.77322911, 145.45974907,
    150.17136399, 154.90745399, 159.66742919, 164.45072752, 169.25681285,
    174.08517319, 178.93531914, 183.80678238
};

// this code was generated by derivations/clustering.py
template<class count_t>
float Clustering<count_t>::LowEntropy::log_partition_function(
        count_t sample_size) const {
    // TODO(fobermeyer) incorporate dataset_size for higher accuracy
    count_t n = sample_size;
    if (n < 48) {
        return log_partition_function_table[n];
    } else {
        float coeff = 0.28269584f;
        float log_z_max = n * fast_log(n);
        return log_z_max * (1.f + coeff * powf(n, -0.75f));
    }
}

// ad hoc approximation,
// see `python derivations/clustering.py dataprob`
// see `python derivations/clustering.py approximations`
template<class count_t>
inline float Clustering<count_t>::LowEntropy::_approximate_dataprob_correction(
        count_t sample_size) const {
    float n = fast_log(sample_size);
    float N = fast_log(dataset_size);
    return 0.061f * n * (n - N) * powf(n + N, 0.75f);
}

template<class count_t>
float Clustering<count_t>::LowEntropy::score_counts(
        const std::vector<count_t> & counts) const {
    float score = 0.0;
    count_t sample_size = 0;
    for (count_t count : counts) {
        sample_size += count;
        if (count > 1) {
            score += count * fast_log(count);
        }
    }
    DIST_ASSERT_LE(sample_size, dataset_size);

    if (sample_size != dataset_size) {
        float log_factor = _approximate_postpred_correction(sample_size);
        score += log_factor * (counts.size() - 1);
        score += _approximate_dataprob_correction(sample_size);
    }
    score -= log_partition_function(sample_size);
    return score;
}

template<class count_t>
std::vector<count_t> Clustering<count_t>::LowEntropy::sample_assignments(
        count_t sample_size,
        rng_t & rng) const {
    DIST_ASSERT_LE(sample_size, dataset_size);

    std::vector<count_t> assignments(sample_size);
    std::vector<count_t> counts;
    std::vector<float> likelihoods;
    counts.reserve(100);
    likelihoods.reserve(100);
    const count_t bogus = 0;
    count_t size = 0;

    for (count_t & assign : assignments) {
        float likelihood_empty = fast_exp(score_add_value(0, bogus, size));
        if (DIST_UNLIKELY(counts.empty()) or counts.back()) {
            counts.push_back(0);
            likelihoods.push_back(likelihood_empty);
        } else {
            likelihoods.back() = likelihood_empty;
        }

        assign = sample_from_likelihoods(rng, likelihoods);
        count_t & count = counts[assign];
        count += 1;
        size += 1;
        float & likelihood = likelihoods[assign];
        float new_likelihood = fast_exp(score_add_value(count, bogus, bogus));
        likelihood = new_likelihood;
    }

    return assignments;
}


// --------------------------------------------------------------------------
// Explicit template instantiation

template struct Clustering<int32_t>;
#if 0
template struct Clustering<int64_t>;
template struct Clustering<uint32_t>;
template struct Clustering<uint64_t>;
#endif

}   // namespace distributions
