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

int sample_discrete (
        rng_t & rng,
        size_t dim,
        const float * probs)
{
    double t = sample_unif01(rng);
    for (size_t i = 0; i < dim - 1; ++i) {
        t -= probs[i];
        if (t < 0) {
            return i;
        }
    }
    return dim - 1;
}

} // namespace distributions
