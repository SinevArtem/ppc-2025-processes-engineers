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

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  int MPI_Allreduce_custom(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

  void performOperation(void *inout, const void *in, int count, MPI_Datatype datatype, MPI_Op op);

  int getTypeSize(MPI_Datatype datatype) const;
};

}  // namespace sinev_a_allreduce
