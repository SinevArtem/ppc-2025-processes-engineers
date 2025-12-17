#include <gtest/gtest.h>

#include <mpi.h>

#include <random>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  int rank_ = 0;

  void SetUp() override {
    // Большой массив для тестирования производительности
    int size = 100000;  // 100 тысяч элементов
    
    // Создаем данные в зависимости от счетчика теста
    static int test_counter = 0;
    test_counter++;
    
    switch (test_counter % 3) {
      case 0: { // int
        std::vector<int> data(size);
        for (int i = 0; i < size; i++) {
          data[i] = i + 1;
        }
        input_data_ = data;
        break;
      }
      case 1: { // float
        std::vector<float> data(size);
        for (int i = 0; i < size; i++) {
          data[i] = static_cast<float>(i + 1) * 0.5f;
        }
        input_data_ = data;
        break;
      }
      case 2: { // double
        std::vector<double> data(size);
        for (int i = 0; i < size; i++) {
          data[i] = static_cast<double>(i + 1) * 0.25;
        }
        input_data_ = data;
        break;
      }
    }
  }

  bool CheckTestOutputData([[maybe_unused]] OutType &output_data) final {
    return  true;
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
