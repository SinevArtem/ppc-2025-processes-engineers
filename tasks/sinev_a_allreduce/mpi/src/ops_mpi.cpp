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
  // Проверяем инициализацию MPI
    int initialized;
    MPI_Initialized(&initialized);
    return initialized == 1;
}

bool SinevAAllreduce::PreProcessingImpl() {
  // Инициализируем MPI, если нужно
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    return true;
}

int SinevAAllreduce::GetTypeSize(MPI_Datatype datatype) const {
  if (datatype == MPI_INT) return sizeof(int);
  if (datatype == MPI_FLOAT) return sizeof(float);
  if (datatype == MPI_DOUBLE) return sizeof(double);
  return 1; // fallback
}

void SinevAAllreduce::PerformOperation(void *inout, const void *in, int count,
                                      MPI_Datatype datatype, MPI_Op op) const {
  if (op == MPI_SUM) {
    if (datatype == MPI_INT) {
      int* out_int = static_cast<int*>(inout);
      const int* in_int = static_cast<const int*>(in);
      for (int i = 0; i < count; ++i) {
        out_int[i] += in_int[i];
      }
    } else if (datatype == MPI_FLOAT) {
      float* out_float = static_cast<float*>(inout);
      const float* in_float = static_cast<const float*>(in);
      for (int i = 0; i < count; ++i) {
        out_float[i] += in_float[i];
      }
    } else if (datatype == MPI_DOUBLE) {
      double* out_double = static_cast<double*>(inout);
      const double* in_double = static_cast<const double*>(in);
      for (int i = 0; i < count; ++i) {
        out_double[i] += in_double[i];
      }
    }
  } else if (op == MPI_MAX) {
    if (datatype == MPI_INT) {
      int* out_int = static_cast<int*>(inout);
      const int* in_int = static_cast<const int*>(in);
      for (int i = 0; i < count; ++i) {
        out_int[i] = std::max(out_int[i], in_int[i]);
      }
    } else if (datatype == MPI_FLOAT) {
      float* out_float = static_cast<float*>(inout);
      const float* in_float = static_cast<const float*>(in);
      for (int i = 0; i < count; ++i) {
        out_float[i] = std::max(out_float[i], in_float[i]);
      }
    } else if (datatype == MPI_DOUBLE) {
      double* out_double = static_cast<double*>(inout);
      const double* in_double = static_cast<const double*>(in);
      for (int i = 0; i < count; ++i) {
        out_double[i] = std::max(out_double[i], in_double[i]);
      }
    }
  } else if (op == MPI_MIN) {
    if (datatype == MPI_INT) {
      int* out_int = static_cast<int*>(inout);
      const int* in_int = static_cast<const int*>(in);
      for (int i = 0; i < count; ++i) {
        out_int[i] = std::min(out_int[i], in_int[i]);
      }
    } else if (datatype == MPI_FLOAT) {
      float* out_float = static_cast<float*>(inout);
      const float* in_float = static_cast<const float*>(in);
      for (int i = 0; i < count; ++i) {
        out_float[i] = std::min(out_float[i], in_float[i]);
      }
    } else if (datatype == MPI_DOUBLE) {
      double* out_double = static_cast<double*>(inout);
      const double* in_double = static_cast<const double*>(in);
      for (int i = 0; i < count; ++i) {
        out_double[i] = std::min(out_double[i], in_double[i]);
      }
    }
  }
}

int SinevAAllreduce::MPI_Allreduce_custom(const void *sendbuf, void *recvbuf, int count,
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  int type_size = GetTypeSize(datatype);
  int total_size = count * type_size;
  
  // Выделяем временный буфер для приема данных
  std::vector<char> temp_buffer(total_size);
  std::vector<char> result_buffer(total_size);
  
  // Копируем исходные данные в буфер результата
  if (sendbuf == MPI_IN_PLACE) {
    // Для MPI_IN_PLACE данные уже находятся в recvbuf
    std::memcpy(result_buffer.data(), recvbuf, total_size);
  } else {
    std::memcpy(result_buffer.data(), sendbuf, total_size);
  }
  
  // Фаза 1: Reduce через двоичное дерево к корню (процесс 0)
  // Используем схему "reduce-scatter" через двоичное дерево
  
  int mask = 1;
  while (mask < size) {
    // Определяем партнера для коммуникации
    int partner = rank ^ mask;
    
    if (partner < size) {
      if ((rank & mask) == 0) {
        // Этот процесс получает данные от партнера и выполняет операцию
        MPI_Recv(temp_buffer.data(), total_size, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
        PerformOperation(result_buffer.data(), temp_buffer.data(), count, datatype, op);
      } else {
        // Этот процесс отправляет данные партнеру
        MPI_Send(result_buffer.data(), total_size, MPI_BYTE, partner, 0, comm);
        break; // После отправки этот процесс завершает фазу reduce
      }
    }
    mask <<= 1; // Увеличиваем маску в 2 раза
  }
  
  // Фаза 2: Broadcast результата от корня ко всем процессам
  // Снова используем двоичное дерево, но теперь для рассылки
  
  if (rank == 0) {
    // Корень копирует результат в выходной буфер
    std::memcpy(recvbuf, result_buffer.data(), total_size);
  }
  
  // Рассылка результата через двоичное дерево
  mask = size >> 1;
  while (mask > 0) {
    if (rank < mask) {
      // Процесс отправляет данные партнеру
      int partner = rank + mask;
      if (partner < size) {
        MPI_Send(recvbuf, total_size, MPI_BYTE, partner, 0, comm);
      }
    } else if (rank < (mask << 1)) {
      // Процесс получает данные от партнера
      int partner = rank - mask;
      MPI_Recv(recvbuf, total_size, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
    }
    mask >>= 1; // Уменьшаем маску в 2 раза
  }
  
  return MPI_SUCCESS;
}

bool SinevAAllreduce::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Получаем данные из варианта
  auto& input_variant = GetInput();
  auto& output_variant = GetOutput();
  
  // Определяем тип данных и выполняем Allreduce
  if (std::holds_alternative<std::vector<int>>(input_variant)) {
    auto& input = std::get<std::vector<int>>(input_variant);
    auto& output = std::get<std::vector<int>>(output_variant);
    
    // Копируем входные данные в выходной буфер для операции
    output = input;
    
    MPI_Allreduce_custom(input.data(), output.data(), input.size(), 
                        MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
  } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
    auto& input = std::get<std::vector<float>>(input_variant);
    auto& output = std::get<std::vector<float>>(output_variant);
    
    output = input;
    
    MPI_Allreduce_custom(input.data(), output.data(), input.size(), 
                        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
  } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
    auto& input = std::get<std::vector<double>>(input_variant);
    auto& output = std::get<std::vector<double>>(output_variant);
    
    output = input;
    
    MPI_Allreduce_custom(input.data(), output.data(), input.size(), 
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  
  return true;
}

bool SinevAAllreduce::PostProcessingImpl() {
  true;
}

}  // namespace sinev_a_allreduce
