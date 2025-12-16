#include <gtest/gtest.h>
#include <stb/stb_image.h>

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

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return 
  }

  InType GetTestInputData() final {
    return 
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SinevAAllreduceFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SinevAAllreduce, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce),
                   ppc::util::AddFuncTask<SinevAAllreduceSEQ, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SinevAAllreduceFuncTests::PrintFuncTestName<SinevAAllreduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SinevAAllreduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sinev_a_allreduce
