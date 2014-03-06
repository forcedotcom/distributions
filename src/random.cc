#include <distributions/random.hpp>

namespace distributions
{

rng_t global_rng;

void sample_dirichlet (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs)
{
    float total = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        if (alphas[i] > 0) {
            total += probs[i] = sample_gamma(rng, alphas[i]);
        } else {
            probs[i] = 0;
        }
    }
    float scale = 1.f / total;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= scale;
    }
}

void sample_dirichlet_safe (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs,
        float min_value)
{
    DIST_ASSERT(min_value >= 0, "bad bound: " << min_value);
    float total = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float alpha = alphas[i] + min_value;
        DIST_ASSERT(alpha > 0, "bad alphas[" << i << "] = " << alpha);
        total += probs[i] = sample_gamma(rng, alpha) + min_value;
    }
    float scale = 1.f / total;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= scale;
    }
}

//----------------------------------------------------------------------------
// Discrete distribution

size_t sample_discrete (
        rng_t & rng,
        size_t dim,
        const float * probs)
{
    float t = sample_unif01(rng);
    for (size_t i = 0; i < dim - 1; ++i) {
        t -= probs[i];
        if (t < 0) {
            return i;
        }
    }
    return dim - 1;
}

size_t sample_from_likelihoods (
        rng_t & rng,
        const std::vector<float> & likelihoods,
        float total_likelihood)
{
    const size_t size = likelihoods.size();

    float t = total_likelihood * sample_unif01(rng);

    for (size_t i = 0; i < size; ++i) {
        t -= likelihoods[i];
        if (t < 0) {
            return i;
        }
    }

    return size - 1;
}

float scores_to_likelihoods (std::vector<float> & scores)
{
    const size_t size = scores.size();
    float * __restrict__ scores_data = scores.data();
    float max_score = vector_max(size, scores_data);

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += scores_data[i] = expf(scores_data[i] - max_score);
    }

    return total;
}

float score_from_scores_overwrite (
        rng_t & rng,
        size_t sample,
        std::vector<float> & scores)
{
    const size_t size = scores.size();
    float * __restrict__ scores_data = scores.data();
    float max_score = vector_max(size, scores_data);

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += expf(scores_data[i] -= max_score);
    }

    if (SYNCHRONIZE_ENTROPY_FOR_UNIT_TESTING) {
        sample_unif01(rng);  // consume entropy to match sampler
    }

    float score = scores_data[sample] - log(total);
    return score;
}

} // namespace distributions
