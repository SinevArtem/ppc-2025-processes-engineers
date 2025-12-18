#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"

#include <cstddef>
#include <cstring>
#include <variant>
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
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized == 1;
}

bool SinevAAllreduce::PreProcessingImpl() {
  return true;
}

int SinevAAllreduce::GetTypeSize(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 1;
}

template <typename T>
static void performSumTemplate(T *out, const T *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}

void SinevAAllreduce::PerformOperation(void *inout, const void *in, int count, MPI_Datatype datatype, MPI_Op op) {
  if (op != MPI_SUM) {
    return;
  }

  if (datatype == MPI_INT) {
    performSumTemplate(static_cast<int *>(inout), static_cast<const int *>(in), count);
  } else if (datatype == MPI_FLOAT) {
    performSumTemplate(static_cast<float *>(inout), static_cast<const float *>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    performSumTemplate(static_cast<double *>(inout), static_cast<const double *>(in), count);
  }
}

int SinevAAllreduce::MpiAllreduceCustom(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                        MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size == 1) {
    int type_size = GetTypeSize(datatype);
    std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count * type_size));
    return 0;
  }

  int type_size = GetTypeSize(datatype);
  int total_bytes = count * type_size;

  std::vector<char> local_buffer(total_bytes);

  // Копируем входные данные
  if (sendbuf == MPI_IN_PLACE) {
    std::memcpy(local_buffer.data(), recvbuf, total_bytes);
  } else {
    std::memcpy(local_buffer.data(), sendbuf, total_bytes);
  }

  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;

    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> recv_buffer(total_bytes);
        MPI_Recv(recv_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);

        PerformOperation(local_buffer.data(), recv_buffer.data(), count, datatype, op);
      } else {
        MPI_Send(local_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }

  if (rank == 0) {
    std::memcpy(recvbuf, local_buffer.data(), total_bytes);

    for (int i = 1; i < size; i++) {
      MPI_Send(recvbuf, count, datatype, i, 1, comm);
    }
  } else {
    MPI_Recv(recvbuf, count, datatype, 0, 1, comm, MPI_STATUS_IGNORE);
  }

  return 0;
}

bool SinevAAllreduce::RunImpl() {
  auto &input_variant = GetInput();
  auto &output_variant = GetOutput();

  try {
    if (std::holds_alternative<std::vector<int>>(input_variant)) {
      auto &input = std::get<std::vector<int>>(input_variant);
      auto &output = std::get<std::vector<int>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
      auto &input = std::get<std::vector<float>>(input_variant);
      auto &output = std::get<std::vector<float>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_FLOAT, MPI_SUM,
                         MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
      auto &input = std::get<std::vector<double>>(input_variant);
      auto &output = std::get<std::vector<double>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD);
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
