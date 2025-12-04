#include <gtest/gtest.h>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SinevAAllreducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SinevAAllreduce, SinevAAllreduceSEQ>(PPC_SETTINGS_sinev_a_allreduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SinevAAllreducePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SinevAAllreducePerfTests, kGtestValues, kPerfTestName);

}  // namespace sinev_a_allreduce
