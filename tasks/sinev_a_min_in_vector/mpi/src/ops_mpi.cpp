#include "sinev_a_min_in_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <limits>
#include <algorithm>

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
  GetOutput() = std::numeric_limits<int>::max();
  return true;
}

bool SinevAMinInVectorMPI::RunImpl() {
  int ProcNum = 0;
  int ProcRank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  std::vector<int> *generalVector = nullptr;
  int generalVectorSize = 0;

  // Только процесс 0 знает исходные данные
  if (ProcRank == 0) {
    generalVector = &GetInput();
    generalVectorSize = generalVector->size();
  }

  MPI_Bcast(&generalVectorSize, 1 , MPI_INT, 0, MPI_COMM_WORLD);

  if (!generalVectorSize) {
    return true;
  }
 

  int blockVector = generalVectorSize / ProcNum;
  int remainder = generalVectorSize % ProcNum; // остаток, если не идеально ровно поделим

  std::vector<int> elemCountInVector(ProcNum);
  std::vector<int> startIndexForLocalVector(ProcNum);

  int offset = 0;
  for (int i = 0; i < ProcNum; i++)
  {
    int needRemeinder = 0;
    if (i < remainder) needRemeinder = 1;

    elemCountInVector[i] = blockVector + needRemeinder;
    startIndexForLocalVector[i] = offset;
    offset += elemCountInVector[i];
  }
  
  
  std::vector<int> localVector(elemCountInVector[ProcRank]); // каждый процесс получает свой фрагмент


  MPI_Scatterv(
    (ProcRank == 0) ? generalVector->data() : nullptr,  
    elemCountInVector.data(),                       
    startIndexForLocalVector.data(),                
    MPI_INT,                                        
    localVector.data(),                             
    elemCountInVector[ProcRank],                    
    MPI_INT,                                        
    0,                                              
    MPI_COMM_WORLD
  );
  
  
  int minNumber = std::numeric_limits<int>::max();
  for (size_t i = 0; i < localVector.size(); i++) {
    if (localVector[i] < minNumber) {
      minNumber = localVector[i];
    }
  }

  int generalMin = std::numeric_limits<int>::max();
  MPI_Allreduce(&minNumber, &generalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = generalMin;

  return true;

}

bool SinevAMinInVectorMPI::PostProcessingImpl() {
  return GetOutput() > std::numeric_limits<int>::min();
}

}  // namespace sinev_a_min_in_vector
