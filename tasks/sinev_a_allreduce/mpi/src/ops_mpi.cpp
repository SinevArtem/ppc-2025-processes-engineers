#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <cstring>
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

int SinevAAllreduce::getTypeSize(MPI_Datatype datatype) {
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
  
  if (size == 1) {
    int type_size = getTypeSize(datatype);
    std::memcpy(recvbuf, sendbuf, count * type_size);
    return 0;
  }
  
  int type_size = getTypeSize(datatype);
  int total_size = count * type_size;
  
  std::vector<char> local_buffer(total_size);
  std::memcpy(local_buffer.data(), sendbuf, total_size);
  
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;
    
    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> recv_buffer(total_size);
        MPI_Recv(recv_buffer.data(), total_size, MPI_BYTE, 
                 partner, 0, comm, MPI_STATUS_IGNORE);
        performOperation(local_buffer.data(), recv_buffer.data(), 
                         count, datatype, op);
      } else {
        MPI_Send(local_buffer.data(), total_size, MPI_BYTE, 
                 partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
  
  if (rank == 0) {
    std::memcpy(recvbuf, local_buffer.data(), total_size);
    
    for (int i = 1; i < size; i++) {
      MPI_Send(local_buffer.data(), count, datatype, i, 1, comm);
    }
  } else {
    MPI_Recv(recvbuf, count, datatype, 0, 1, comm, MPI_STATUS_IGNORE);
  }
  
  return 0;
}

bool SinevAAllreduce::RunImpl() {
  auto& input_variant = GetInput();
  auto& output_variant = GetOutput();
  
  try {
    if (std::holds_alternative<std::vector<int>>(input_variant)) {
      auto& input = std::get<std::vector<int>>(input_variant);
      auto& output = std::get<std::vector<int>>(output_variant);
      
      if (output.size() != input.size()) {
        output.resize(input.size());
      }
      
      MPI_Allreduce_custom(input.data(), output.data(), 
                          static_cast<int>(input.size()), 
                          MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      
    } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
      auto& input = std::get<std::vector<float>>(input_variant);
      auto& output = std::get<std::vector<float>>(output_variant);
      
      if (output.size() != input.size()) {
        output.resize(input.size());
      }
      
      MPI_Allreduce_custom(input.data(), output.data(), 
                          static_cast<int>(input.size()), 
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      
    } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
      auto& input = std::get<std::vector<double>>(input_variant);
      auto& output = std::get<std::vector<double>>(output_variant);
      
      if (output.size() != input.size()) {
        output.resize(input.size());
      }
      
      MPI_Allreduce_custom(input.data(), output.data(), 
                          static_cast<int>(input.size()), 
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    
    return true;
  } catch (...) {
    return false;
  }
}

bool SinevAAllreduce::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce