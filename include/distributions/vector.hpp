#pragma once

#include <distributions/aligned_allocator.hpp>
#include <vector>

namespace distributions
{

typedef std::vector<float, aligned_allocator<float>> FloatVector;

} // namespace distributions
