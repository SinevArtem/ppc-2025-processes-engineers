#include <gtest/gtest.h>

#include "sinev_a_min_in_vector/common/include/common.hpp"
#include "sinev_a_min_in_vector/mpi/include/ops_mpi.hpp"
#include "sinev_a_min_in_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sinev_a_min_in_vector {

class SinevAMinInVectorPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  int realMin;
  InType input_data_{};

  void SetUp() override {
    int size = 100000000;  
    input_data_.resize(size);
    
    std::fill(input_data_.begin(), input_data_.end(), 100);
    input_data_[size / 2] = -1;  
    realMin = -1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == realMin;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SinevAMinInVectorPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SinevAMinInVectorMPI, SinevAMinInVectorSEQ>(PPC_SETTINGS_sinev_a_min_in_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SinevAMinInVectorPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SinevAMinInVectorPerfTests, kGtestValues, kPerfTestName);

}  // namespace sinev_a_min_in_vector
