#include <distributions/random.hpp>

namespace distributions
{

rng_t global_rng;

void sample_dirichlet (
        size_t dim,
        const float * alphas,
        float * ps,
        rng_t & rng)
{
    typedef gamma_distribution_t::param_type param_t;
    gamma_distribution_t sampler;

    double total = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        total += ps[i] = sample_gamma(alphas[i], 1.0, rng);
    }

    double scale = 1.0 / total;
    for (size_t i = 0; i < dim; ++i) {
        ps[i] *= scale;
    }
}

int sample_discrete (
        size_t dim,
        const float * ps,
        rng_t & rng)
{
    float t = sample_unif01(rng);
    for (size_t i = 0; i < dim - 1; ++i) {
        t -= ps[i];
        if (t < 0) {
            return i;
        }
    }
    return dim - 1;
}

} // namespace distributions
