#pragma once

namespace distributions
{

template<class Model>
struct SharedMixin
{
    bool add_value (const typename Model::Value &) { return false; }
    bool remove_value (const typename Model::Value &) { return false; }
};

} // namespace distributions
