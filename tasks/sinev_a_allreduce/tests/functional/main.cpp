#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <cmath>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <mpi.h>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "size_" + std::to_string(std::get<0>(test_param)) + "_type_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int vector_size = std::get<0>(params);
    std::string data_type = std::get<1>(params);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (data_type == "int") {
      std::vector<int> vec(vector_size, rank + 1);  // все элементы = rank+1
      input_data_ = vec;
    } else if (data_type == "float") {
      std::vector<float> vec(vector_size, static_cast<float>(rank + 1));
      input_data_ = vec;
    } else if (data_type == "double") {
      std::vector<double> vec(vector_size, static_cast<double>(rank + 1));
      input_data_ = vec;
    }
    
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int expected_sum = 0;
    for (int i = 0; i < size; i++) {
      expected_sum += (i + 1);
    }


    // Проверяем в зависимости от типа данных
    if (std::holds_alternative<std::vector<int>>(output_data)) {
      auto& vec = std::get<std::vector<int>>(output_data);
      return !vec.empty() && vec[0] == expected_sum;
    } else if (std::holds_alternative<std::vector<float>>(output_data)) {
      auto& vec = std::get<std::vector<float>>(output_data);
      return !vec.empty() && std::fabs(vec[0] - expected_sum) < 1e-6f;
    } else if (std::holds_alternative<std::vector<double>>(output_data)) {
      auto& vec = std::get<std::vector<double>>(output_data);
      return !vec.empty() && std::fabs(vec[0] - expected_sum) < 1e-9;
    }
    
    return false;  // не распознали тип
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SinevAAllreduceFuncTests, VectorAllreduceTests) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: (размер_вектора, тип_данных)
// Нужно исправить "3", "5", "7" на реальные типы
const std::array<TestType, 6> kTestParam = {
  std::make_tuple(10, "int"),     // 10 элементов int
  std::make_tuple(100, "int"),    // 100 элементов int
  std::make_tuple(1000, "int"),   // 1000 элементов int
  std::make_tuple(10, "float"),   // 10 элементов float
  std::make_tuple(100, "float"),  // 100 элементов float
  std::make_tuple(10, "double")   // 10 элементов double
};


const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SinevAAllreduce, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce),
                   ppc::util::AddFuncTask<SinevAAllreduceSEQ, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SinevAAllreduceFuncTests::PrintFuncTestName<SinevAAllreduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAllreduceTests, SinevAAllreduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sinev_a_allreduce
