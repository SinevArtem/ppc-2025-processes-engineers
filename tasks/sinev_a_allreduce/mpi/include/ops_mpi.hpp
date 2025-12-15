#pragma once

#include "sinev_a_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduce : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SinevAAllreduce(const InType &in);

  int MPI_Allreduce_custom(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  // Вспомогательные функции для ALLREDUCE
  int GetTypeSize(MPI_Datatype datatype) const;
  void PerformOperation(void *inout, const void *in, int count,
                       MPI_Datatype datatype, MPI_Op op) const;
};

}  // namespace sinev_a_allreduce
