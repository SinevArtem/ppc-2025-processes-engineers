#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sinev_a_min_in_vector {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sinev_a_min_in_vector

// sinev_a_min_in_vector