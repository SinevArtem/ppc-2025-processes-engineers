#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sinev_a_allreduce {

SinevAAllreduce::SinevAAllreduce(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}



bool SinevAAllreduce::ValidationImpl() {
  int initialized;
  MPI_Initialized(&initialized);
  return initialized == 1;
}

bool SinevAAllreduce::PreProcessingImpl() {
  return true;
}
  
int SinevAAllreduce::getTypeSize(MPI_Datatype datatype) const {
  if (datatype == MPI_INT) return sizeof(int);
  if (datatype == MPI_FLOAT) return sizeof(float);
  if (datatype == MPI_DOUBLE) return sizeof(double);
  return 1;
}

template<typename T>
void performSumTemplate(T* out, const T* in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}

void SinevAAllreduce::performOperation(void *inout, const void *in, int count,
                                      MPI_Datatype datatype, MPI_Op op) {
  if (op != MPI_SUM) return;
  
  if (datatype == MPI_INT) {
    performSumTemplate(static_cast<int*>(inout), static_cast<const int*>(in), count);
  } else if (datatype == MPI_FLOAT) {
    performSumTemplate(static_cast<float*>(inout), static_cast<const float*>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    performSumTemplate(static_cast<double*>(inout), static_cast<const double*>(in), count);
  }
}

int SinevAAllreduce::MPI_Allreduce_custom(const void *sendbuf, void *recvbuf, int count,
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  int type_size = getTypeSize(datatype);
  int total_size = count * type_size;
  
  // Локальный буфер
  std::vector<char> buffer(total_size);
  std::memcpy(buffer.data(), sendbuf, total_size);
  
  // Фаза 1: Reduce через двоичное дерево
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;
    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> temp(total_size);
        MPI_Recv(temp.data(), total_size, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
        performOperation(buffer.data(), temp.data(), count, datatype, op);
      } else {
        MPI_Send(buffer.data(), total_size, MPI_BYTE, partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
  
  // Фаза 2: Broadcast через обратное дерево
  mask = size >> 1;
  while (mask > 0) {
    int partner = rank ^ mask;
    if (partner < size) {
      if ((rank & mask) == 0) {
        MPI_Send(buffer.data(), total_size, MPI_BYTE, partner, 1, comm);
      } else {
        MPI_Recv(buffer.data(), total_size, MPI_BYTE, partner, 1, comm, MPI_STATUS_IGNORE);
      }
    }
    mask >>= 1;
  }
  
  std::memcpy(recvbuf, buffer.data(), total_size);
  return MPI_SUCCESS;
}

bool SinevAAllreduce::RunImpl() {
  auto& input_variant = GetInput();
  auto& output_variant = GetOutput();

  // Общая лямбда для выполнения Allreduce
  auto executeAllreduce = [this](auto& input, auto& output, MPI_Datatype type) {
    output = input; // Копируем вход в выход
    MPI_Allreduce_custom(input.data(), output.data(), input.size(), 
                         type, MPI_SUM, MPI_COMM_WORLD);
  };
  
  // Проверяем тип данных и выполняем Allreduce
  if (std::holds_alternative<std::vector<int>>(input_variant)) {
    auto& input = std::get<std::vector<int>>(input_variant);
    auto& output = std::get<std::vector<int>>(output_variant);
    executeAllreduce(input, output, MPI_INT);
    
  } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
    auto& input = std::get<std::vector<float>>(input_variant);
    auto& output = std::get<std::vector<float>>(output_variant);
    executeAllreduce(input, output, MPI_FLOAT);
    
  } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
    auto& input = std::get<std::vector<double>>(input_variant);
    auto& output = std::get<std::vector<double>>(output_variant);
    executeAllreduce(input, output, MPI_DOUBLE);
  }
  
  return true;
}

bool SinevAAllreduce::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
