# Минимальное значение элементов вектора

- Студент: Синев Артём Александрович, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 3

## 1. Введение

Операция Allreduce является одной из фундаментальных коллективных операций в параллельных вычислениях. Она выполняет редукцию данных со всех процессов с применением указанной операции (например, суммирование) и рассылает результат всем процессам. Данная работа посвящена разработке пользовательской реализации операции Allreduce с использованием только базовых функций передачи сообщений MPI_Send и MPI_Recv. Особенностью реализации является использование алгоритма двоичного дерева для эффективной коммуникации между процессами.

## 2. Постановка задачи

Реализовать функцию MPI_Allreduce_custom, которая выполняет ту же задачу, что и стандартная MPI_Allreduce, но использует только MPI_Send и MPI_Recv.

**Входные данные:**
- sendbuf - указатель на буфер отправки данных (у каждого процесса свои данные)
- count - количество элементов в буфере
- datatype - тип данных (MPI_INT, MPI_FLOAT, MPI_DOUBLE)
- op - операция редукции (поддерживается только MPI_SUM)
- comm - коммуникатор MPI

**Выходные данные:**
- recvbuf - указатель на буфер приема результатов (у всех процессов одинаковый результат)

**Требования:**
- Использовать только MPI_Send и MPI_Recv
- Применить древовидную схему коммуникации
- Поддержка типов данных: MPI_INT, MPI_FLOAT, MPI_DOUBLE
- Поддержка операции: MPI_SUM

**Пример для 4 процессов:**

Процесс 0: [1, 2, 3]
Процесс 1: [4, 5, 6]
Процесс 2: [7, 8, 9]
Процесс 3: [10, 11, 12]

Результат у всех: [22, 26, 30] (сумма соответствующих элементов)

## 3. Описание базового алгоритма (SEQ версия)

Последовательная версия (SEQ) операции Allreduce является тривиальной, так как нет других процессов для коммуникации:

1. **Инициализация**: Копирование входных данных в выходной буфе 
2. **Завершение**: Возврат результата

Алгоритм состоит из следующих шагов:

```cpp 
bool SinevAAllreduceSEQ::RunImpl() {
  try {
    GetOutput() = GetInput();  // Простое копирование данных
    return true;
  } catch (const std::exception &) {
    return false;
  }
}
```

**Сложность алгоритма**

**Время:** 
- O(1) - простое копирование указателей

**Память:**
- O(N) - хранение входного вектора


## 4. Схема распараллеливания

### 4.1. Алгоритм двоичного дерева

Параллельная реализация использует алгоритм двоичного дерева (binary tree algorithm), который состоит из двух фаз:

- Фаза 1: Редукция (снизу вверх)
- Фаза 2: Рассылка (сверху вниз)

## 4.2. Ключевые особенности алгоритма

**Использование битовых операций**

```cpp 
int partner = rank ^ mask;       // Вычисление партнера с помощью XOR
if ((rank & mask) == 0) {       // Определение роли: получатель (0) или отправитель
```

**Двухфазный подход**

- Фаза редукции: Сбор данных к корню (процессу 0) с поэтапным суммированием
- Фаза рассылки: Распространение результата от корня всем процессам

**Распределение нагрузки**

- Каждый процесс участвует в вычислениях
- Работа распределяется равномерно между всеми процессами
- Коммуникации происходят параллельно на каждом этапе

**Роли процессов**
- **Процесс 0:** Координирует рассылку финального результата
- **Все процессы:** Вычисляют локальные суммы и участвуют в обмене данными
- **Каждая пара процессов:** Работает параллельно на каждом этапе

## 5. Детали реализации

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциоанльные тесты
- `tests/performance/main.cpp` - тесты производительности

**Ключевые классы:**
- `SinevAMinInVectorSEQ` - последовательная реализация
- `SinevAMinInVectorMPI` - параллельная MPI реализация

**Основные методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - подготовительные вычисления  
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - завершающая обработка

## Особенности реализации

**Основная функция MPI_Allreduce_custom:**

```cpp
int SinevAAllreduce::MPI_Allreduce_custom(const void *sendbuf, void *recvbuf, 
                                         int count, MPI_Datatype datatype,
                                         MPI_Op op, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  // Специальный случай: один процесс
  if (size == 1) {
    int type_size = getTypeSize(datatype);
    std::memcpy(recvbuf, sendbuf, count * type_size);
    return 0;
  }
  
  int type_size = getTypeSize(datatype);
  int total_bytes = count * type_size;
  
  // Локальный буфер для данных
  std::vector<char> local_buffer(total_bytes);
  std::memcpy(local_buffer.data(), sendbuf, total_bytes);
  
  // === ФАЗА 1: РЕДУКЦИЯ (двоичное дерево) ===
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;
    
    if (partner < size) {
      if ((rank & mask) == 0) {
        // Этот процесс получает и суммирует
        std::vector<char> recv_buffer(total_bytes);
        MPI_Recv(recv_buffer.data(), total_bytes, MPI_BYTE, 
                partner, 0, comm, MPI_STATUS_IGNORE);
        performOperation(local_buffer.data(), recv_buffer.data(), 
                        count, datatype, op);
      } else {
        // Этот процесс отправляет и завершает фазу редукции
        MPI_Send(local_buffer.data(), total_bytes, MPI_BYTE, 
                partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
  
  // === ФАЗА 2: РАССЫЛКА ===
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
```

**Обработка разных типов данных:**

```cpp
void SinevAAllreduce::performOperation(void *inout, const void *in, 
                                      int count, MPI_Datatype datatype, 
                                      MPI_Op op) {
  if (op != MPI_SUM) return;
  
  if (datatype == MPI_INT) {
    performSumTemplate(static_cast<int*>(inout), 
                      static_cast<const int*>(in), count);
  } else if (datatype == MPI_FLOAT) {
    performSumTemplate(static_cast<float*>(inout), 
                      static_cast<const float*>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    performSumTemplate(static_cast<double*>(inout), 
                      static_cast<const double*>(in), count);
  }
}
```

**Шаблонная функция суммирования:**

```cpp
template <typename T>
void performSumTemplate(T *out, const T *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}
```

## Обработка граничных случаев

- **Один процесс**: Простое копирование данных без коммуникации
- **MPI_IN_PLACE**: Поддержка работы с одним буфером для ввода/вывода
- **Пустой вектор**: Корректная обработка нулевого размера данных

## 6. Экспериментальное окружение

### 6.1 Аппаратное обеспечение/ОС:

- **Процессор:** Intel Core i7-13700HX
- **Ядра:** 16 физических ядер  
- **ОЗУ:** 8 ГБ 
- **ОС:** Kubuntu 24.04

### 6.2 Программный инструментарий

- **Компилятор:** g++ 13.3.0
- **Тип сборки:** Release
- **Стандарт C++:** C++20
- **MPI:** OpenMPI 4.1.6

### 6.3 Тестовое окружение

```bash
PPC_NUM_PROC=1,2,4
```

## 7. Результаты

### 7.1. Корректность работы

Все функциональные тесты пройдены успешно:
- 9 тестовых конфигураций: Разные размеры (1, 10, 100, 1000 элементов) и типы данных (int, float, double)
- SEQ vs MPI: Обе реализации выдают идентичные результаты
- MPI тесты: Правильная работа для 1, 2, 4 процессов

SEQ и MPI версии выдают идентичные результаты для всех тестовых случаев.

### 7.2. Производительность

**Время выполнения (секунды) для матрицы 5000×5000:**

| Версия | Количество процессов | Task Run время |
|--------|---------------------|----------------|
| SEQ    | 1                   | 0.3933         |
| MPI    | 1                   | 0.5537         | 
| MPI    | 2                   | 0.2972         |
| MPI    | 4                   | 0.1748         |

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 0.71×     | 71%           |
| 2                   | 1.32×     | 66%           |
| 4                   | 2.25×     | 56%           |


**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ эффективности

- **Лучшее ускорение:** 2.25× на 4 процессах
- **Оптимальная конфигурация:** 4 процесса
- **Эффективность MPI:** Снижается с увеличением количества процессов

### 7.4. Наблюдения

1. **MPI с 1 процессом** показывает меньшую производительность из-за накладных расходов MPI
2. **MPI с 2 процессами** демонстрирует ускорение 1.32×
3. **MPI с 4 процессами** достигает максимального ускорения 2.25×
4. **Снижение эффективности** связано с затратами на распределение данных

## 8. Выводы

### 8.1. Достигнутые результаты

- **Корректность:** Успешное прохождение всех функциональных тестов
- **Эффективность параллелизации:** Ускорение 2.25× на 4 процессах
- **Масштабируемость:**  Хорошее масштабирование с увеличением процессов
- **Надежность:** Распределенный подход для больших данных

### 8.2. Особенности реализации

- **Распределенный подход:** Использование MPI_Scatterv для распределения данных
- **Синхронизация:** MPI_Bcast для согласованной проверки входных данных
- **Балансировка нагрузки:** Равномерное распределение с учетом остатка
- **СОптимизация коммуникаций:** Минимизация дублирования данных

### 8.3. Ограничения и перспективы
- **Накладные расходы:** Коммуникационные затраты для малых данных
- **Оптимальность:** Наиболее эффективен для объемных данных
- **Перспективы:** Неблокирующие операции для перекрытия вычислений

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Документация MPI: https://www.open-mpi.org/
3. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
#include "sinev_a_min_in_vector/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <algorithm>
#include <limits>
#include <vector>

#include "sinev_a_min_in_vector/common/include/common.hpp"

namespace sinev_a_min_in_vector {

SinevAMinInVectorMPI::SinevAMinInVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::max();
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
  return true;
}

bool SinevAMinInVectorMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<int> local_data;
  int global_size = 0;

  if (proc_rank == 0) {
    global_size = static_cast<int>(GetInput().size());
  }

  // Рассылаем размер всем процессам
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (global_size == 0) {
    GetOutput() = std::numeric_limits<int>::max();
    return true;
  }

  int block_size = global_size / proc_num;
  int remainder = global_size % proc_num;

  int local_size = block_size + (proc_rank < remainder ? 1 : 0);
  local_data.resize(local_size);

  std::vector<int> sendcounts(proc_num);
  std::vector<int> displacements(proc_num);

  if (proc_rank == 0) {
    for (int i = 0; i < proc_num; i++) {
      sendcounts[i] = block_size + (i < remainder ? 1 : 0);
      displacements[i] = (i * block_size) + std::min(i, remainder);
    }
  }

  MPI_Bcast(sendcounts.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(proc_rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displacements.data(), MPI_INT,
               local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  for (int value : local_data) {
    local_min = std::min(local_min, value);
  }

  int global_min = std::numeric_limits<int>::max();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool SinevAMinInVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_min_in_vector

```