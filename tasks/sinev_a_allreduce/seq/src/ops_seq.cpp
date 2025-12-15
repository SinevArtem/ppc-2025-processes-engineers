#include "sinev_a_allreduce/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {

SinevAAllreduceSEQ::SinevAAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SinevAAllreduceSEQ::ValidationImpl() {
  return true;
}

bool SinevAAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool SinevAAllreduceSEQ::RunImpl() {
  return true;
}

bool SinevAAllreduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
