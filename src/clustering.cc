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
// Pitman-Yor Model

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
// Low-Entropy Model

// this code generated by derivations/clustering.py
static const float cluster_normalizing_scores[48] =
{
    0.00000000f, 0.00000000f, 1.60943791f, 3.68887945f,
    6.07993320f, 8.70549682f, 11.51947398f, 14.49108422f,
    17.59827611f, 20.82445752f, 24.15668300f, 27.58456586f,
    31.09958507f, 34.69462231f, 38.36364086f, 42.10145572f,
    45.90356476f, 49.76602176f, 53.68533918f, 57.65841234f,
    61.68245958f, 65.75497413f, 69.87368527f, 74.03652635f,
    78.24160846f, 82.48719834f, 86.77169993f, 91.09363859f,
    95.45164780f, 99.84445762f, 104.27088480f, 108.72982416f,
    113.22024112f, 117.74116515f, 122.29168392f, 126.87093829f,
    131.47811772f, 136.11245629f, 140.77322911f, 145.45974907f,
    150.17136399f, 154.90745399f, 159.66742919f, 164.45072752f,
    169.25681285f, 174.08517319f, 178.93531914f, 183.80678238f
};

// this code generated by derivations/clustering.py
template<class count_t>
inline float cluster_normalizing_score (count_t sample_size)
{
    // FIXME incorporate dataset_size
    count_t n = sample_size;
    if (n < 48) {
        return cluster_normalizing_scores[n];
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
inline float approximate_dataprob_correction(
        count_t sample_size,
        count_t dataset_size)
{
    float n = fast_log(sample_size);
    float N = fast_log(dataset_size);
    return 0.061f * n * (n - N) * powf(n + N, 0.75f);
}

template<class count_t>
float Clustering<count_t>::LowEntropy::score_counts (
        const std::vector<count_t> & counts) const
{
    float score = 0.0;
    count_t sample_size = 0;
    for (count_t count : counts) {
        sample_size += count;
        if (count > 1) {
            score += count * fast_log(count);
        }
    }
    DIST_ASSERT(
        sample_size <= dataset_size,
        "sample_size = " << sample_size <<
        ", dataset_size = " << dataset_size);

    if (sample_size != dataset_size) {
        float log_factor =
            approximate_postpred_correction(sample_size);
        score += log_factor * (counts.size() - 1);
        score += approximate_dataprob_correction(sample_size, dataset_size);
    }
    score -= cluster_normalizing_score(sample_size);
    return score;
}

template<class count_t>
std::vector<count_t> Clustering<count_t>::LowEntropy::sample_assignments (
        count_t sample_size,
        rng_t & rng) const
{
    DIST_ASSERT(
        sample_size <= dataset_size,
        "sample_size = " << sample_size <<
        ", dataset_size = " << dataset_size);

    std::vector<count_t> assignments(sample_size);
    std::vector<count_t> counts;
    std::vector<float> likelihoods;
    counts.reserve(100);
    likelihoods.reserve(100);
    double total = 0;
    count_t size = 1;
    const count_t unused = 0;

    for (count_t & assign : assignments) {

        float likelihood_empty =
            expf(score_add_value(0, unused, size));
        if (counts.empty() or counts.back()) {
            counts.push_back(0);
            likelihoods.push_back(likelihood_empty);
            total += likelihood_empty;
        } else {
            float & likelihood = likelihoods.back();
            total += likelihood_empty - likelihood;
            likelihood = likelihood_empty;
        }

        //assign = sample_from_likelihoods(rng, likelihoods, total);
        assign = sample_from_likelihoods(rng, likelihoods);
        count_t & count = counts[assign];
        count += 1;
        size += 1;
        float & likelihood = likelihoods[assign];
        float new_likelihood =
            expf(score_add_value(count, unused, size));
        total += new_likelihood - likelihood;
        likelihood = new_likelihood;
    }

    return assignments;
}


//----------------------------------------------------------------------------
// Explicit template instantiation

template struct Clustering<int>;

} // namespace distributions
