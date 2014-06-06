#pragma once

#include <distributions/random_fwd.hpp>
#include <distributions/mixture.hpp>

namespace distributions
{

template<class Model_>
struct SharedMixin
{
    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Group Group;

    void add_value (const Value &, rng_t &) {}
    void remove_value (const Value &, rng_t &) {}
};

template<class Model_>
struct GroupMixin
{
    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
};

template<class Model_>
struct VectorizedScorerMixin
{
    typedef Model_ Model;
    typedef typename Model::Value Value;
    typedef typename Model::Shared Shared;
    typedef typename Model::Group Group;
    typedef MixtureSlave<Shared> Slave;
};

} // namespace distributions
