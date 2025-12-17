#include "sinev_a_allreduce/seq/include/ops_seq.hpp"

#include <mpi.h>
#include <numeric>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {


namespace {

// Вспомогательная функция для выполнения MPI_Allreduce
template <typename T>
void PerformMPIAllreduce(const std::vector<T>& input, std::vector<T>& output, 
                         MPI_Datatype mpi_type) {
  // Убеждаемся, что выходной вектор правильного размера
  if (output.size() != input.size()) {
    output.resize(input.size());
  }
  
  // Выполняем MPI_Allreduce для суммирования
  if (!input.empty()) {
    MPI_Allreduce(
      input.data(),        // send buffer (исходные данные)
      output.data(),       // receive buffer (результат суммирования)
      static_cast<int>(input.size()),  // количество элементов
      mpi_type,            // MPI тип данных
      MPI_SUM,             // операция - СУММИРОВАНИЕ
      MPI_COMM_WORLD       // коммуникатор
    );
  }
}

}  // namespace

SinevAAllreduceSEQ::SinevAAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}



bool SinevAAllreduceSEQ::ValidationImpl() {
  int initialized;
  MPI_Initialized(&initialized);
  return initialized == 1;
}

bool SinevAAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool SinevAAllreduceSEQ::RunImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();
  
  // Обрабатываем каждый тип данных
  if (std::holds_alternative<std::vector<int>>(input)) {
    auto& in_vec = std::get<std::vector<int>>(input);
    auto& out_vec = std::get<std::vector<int>>(output);
    PerformMPIAllreduce(in_vec, out_vec, MPI_INT);
    
  } else if (std::holds_alternative<std::vector<float>>(input)) {
    auto& in_vec = std::get<std::vector<float>>(input);
    auto& out_vec = std::get<std::vector<float>>(output);
    PerformMPIAllreduce(in_vec, out_vec, MPI_FLOAT);
    
  } else if (std::holds_alternative<std::vector<double>>(input)) {
    auto& in_vec = std::get<std::vector<double>>(input);
    auto& out_vec = std::get<std::vector<double>>(output);
    PerformMPIAllreduce(in_vec, out_vec, MPI_DOUBLE);
  }
  
  return true;
}

bool SinevAAllreduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
