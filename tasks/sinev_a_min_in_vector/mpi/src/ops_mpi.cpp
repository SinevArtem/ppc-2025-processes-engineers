#include "sinev_a_min_in_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "sinev_a_min_in_vector/common/include/common.hpp"

namespace sinev_a_min_in_vector {

SinevAMinInVectorMPI::SinevAMinInVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SinevAMinInVectorMPI::ValidationImpl() {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  bool is_valid = true;

  if (proc_rank == 0) {
    is_valid = !GetInput().empty();
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  return is_valid;
}

bool SinevAMinInVectorMPI::PreProcessingImpl() {
  GetOutput() = INT_MAX;
  return true;
}

bool SinevAMinInVectorMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<int> &my_vector = GetInput();

  uint64_t global_size = my_vector.size();

  MPI_Bcast(&global_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (global_size == 0) {
    GetOutput() = INT_MAX;
    return true;
  }

  // Проверка на переполнение
  if (global_size > INT_MAX) {
    return false;
  }

  int min_number = INT_MAX;

  int block_size = global_size / proc_num;
  int remainder = global_size % proc_num;

  int start_index = (proc_rank * block_size) + std::min(proc_rank, remainder);
  int end_index = start_index + block_size + (proc_rank < remainder ? 1 : 0);

  if (proc_rank == 0) {
    for (int i = start_index; i < end_index; i++) {
      min_number = std::min(my_vector[i], min_number);
    }
  } else if (end_index > start_index) {
    for (int i = start_index; i < end_index; i++) {
      min_number = std::min(my_vector[i], min_number);
    }
  }

  for (int i = start_index; i < end_index; i++) {
    min_number = std::min(my_vector[i], min_number);
  }

  int general_min = INT_MAX;
  MPI_Allreduce(&min_number, &general_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = general_min;
  return true;
}

bool SinevAMinInVectorMPI::PostProcessingImpl() {
  return GetOutput() > INT_MIN;
}

}  // namespace sinev_a_min_in_vector
