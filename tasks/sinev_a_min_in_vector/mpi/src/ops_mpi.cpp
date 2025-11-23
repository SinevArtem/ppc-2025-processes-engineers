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
  return !GetInput().empty();
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
  int min_number = INT_MAX;

  int block_vector = static_cast<int>(my_vector.size()) / proc_num;
  int remainder = static_cast<int>(my_vector.size()) % proc_num;

  int start_index = (proc_rank * block_vector) + std::min(proc_rank, remainder);
  int end_index = start_index + block_vector + (proc_rank < remainder ? 1 : 0);

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
