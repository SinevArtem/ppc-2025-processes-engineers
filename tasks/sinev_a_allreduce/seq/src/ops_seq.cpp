#include "sinev_a_allreduce/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {


namespace {

template <typename T, MPI_Datatype MpiType>
bool ProcessAllreduceWithMPI(const InType &input, OutType &output) {
  auto &input_vec = std::get<std::vector<T>>(input);
  auto &output_vec = std::get<std::vector<T>>(output);

  // Проверяем размеры
  if (output_vec.size() != input_vec.size()) {
    output_vec.resize(input_vec.size());
  }

  // Используем оригинальный MPI_Allreduce
  if (!input_vec.empty()) {
    // Создаем временную копию для sendbuf (чтобы не менять исходные данные)
    std::vector<T> temp = input_vec;
    
    MPI_Allreduce(
      temp.data(),          // send buffer (временная копия)
      output_vec.data(),    // receive buffer
      static_cast<int>(input_vec.size()),  // count
      MpiType,              // MPI data type
      MPI_SUM,              // operation (сумма)
      MPI_COMM_WORLD        // communicator
    );
  }
  
  return true;
}

}  // namespace

SinevAAllreduceSEQ::SinevAAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool SinevAAllreduceSEQ::ValidationImpl() {
  try {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool SinevAAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool SinevAAllreduceSEQ::RunImpl() {
  try {
    GetOutput() = GetInput(); 
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool SinevAAllreduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
