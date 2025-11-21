#include "sinev_a_min_in_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "sinev_a_min_in_vector/common/include/common.hpp"
#include "util/include/util.hpp"

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
  int ProcNum;
  int ProcRank;

  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  std::vector<int> &myVector = GetInput();
  int minNumber = INT_MAX;

  int blockVector = myVector.size() / ProcNum;
  int remainder = myVector.size() % ProcNum;  // остаток, если не идеально ровно поделим

  int startIndex = ProcRank * blockVector + std::min(ProcRank, remainder);
  int endIndex = startIndex + blockVector + (ProcRank < remainder ? 1 : 0);

  for (int i = startIndex; i < endIndex; i++) {
    if (myVector[i] < minNumber) {
      minNumber = myVector[i];
    }
  }

  int generalMin = INT_MAX;
  MPI_Allreduce(&minNumber, &generalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = generalMin;

  return true;
}

bool SinevAMinInVectorMPI::PostProcessingImpl() {
  return GetOutput() > INT_MIN;
}

}  // namespace sinev_a_min_in_vector
