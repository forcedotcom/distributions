#include <iostream>
#include <cstdio>
#include <distributions/random.hpp>
#include <distributions/clustering.hpp>
#include <distributions/timers.hpp>

using namespace distributions;

size_t speedtest (size_t size, size_t iters, float alpha, float d)
{
    Clustering<int>::PitmanYor model;
    model.alpha = alpha;
    model.d = d;

    rng_t rng;

    int64_t time = -current_time_us();

    size_t bogus = 0;
    for (int i = 0; i < iters; ++i) {
        model.sample_assignments(size, rng);
    }

    time += current_time_us();

    double time_sec = time * 1e-6;
    double samples_per_sec = iters / time_sec;
    std::cout << size << '\t' << samples_per_sec << '\n';

    return bogus;
}

int main (int argc, char ** argv)
{
    float alpha = (argc > 1) ? atof(argv[1]) : 1.0f;
    float d = (argc > 2) ? atof(argv[2]) : 0.0f;

    std::cout << "size" << '\t' << "samples_per_sec";
    std::cout << " (alpha = " << alpha << ", d = " << d << ")\n";

    int min_exponent = 10;
    int max_exponent = 20;
    for (int i = min_exponent; i <= max_exponent; ++i) {
        int size = 1 << i;
        int iters = 4 << (max_exponent - i);
        speedtest(size, iters, alpha, d);
    }

    return 0;
}
