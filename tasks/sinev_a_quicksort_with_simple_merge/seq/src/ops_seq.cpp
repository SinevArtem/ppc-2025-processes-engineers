#include "sinev_a_quicksort_with_simple_merge/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

SinevAQuicksortWithSimpleMergeSEQ::SinevAQuicksortWithSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}



bool SinevAQuicksortWithSimpleMergeSEQ::ValidationImpl() {
  if (GetInput().empty()) return false;
  return true;
}

bool SinevAQuicksortWithSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

int SinevAQuicksortWithSimpleMergeSEQ::Partition(std::vector<int>& arr, int left, int right) {
  int pivot_index = left + (right - left) / 2;
  int pivot_value = arr[pivot_index];

  std::swap(arr[pivot_index], arr[left]);

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) {
      i++;
    }
    
    while (i <= j && arr[j] > pivot_value) {
      j--;
    }

    if (i < j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
    else {
      break;
    }
  }

  std::swap(arr[left], arr[j]);

  return j;
}

void SinevAQuicksortWithSimpleMergeSEQ::SimpleMerge(std::vector<int>& arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);

  int i = left;     
  int j = mid + 1;   
  int k = 0;  // Индекс для временного массива

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k] = arr[i];
      i++;
    } else {
      temp[k] = arr[j];
      j++;
    }
    k++;
  }

  while (i <= mid) {
    temp[k] = arr[i];
    i++;
    k++;
  }

  while (j <= right) {
    temp[k] = arr[j];
    j++;
    k++;
  }

  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void SinevAQuicksortWithSimpleMergeSEQ::QuickSortWithSimpleMerge(std::vector<int>& arr, int left, int right) {
  if (left>=right) {
    return;
  }

  int pivot_index = Partition(arr, left, right);

  QuickSortWithSimpleMerge(arr, left, pivot_index - 1);

  QuickSortWithSimpleMerge(arr, pivot_index+1, right);

  SimpleMerge(arr, left, pivot_index, right);

}

bool SinevAQuicksortWithSimpleMergeSEQ::RunImpl() {
  QuickSortWithSimpleMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);

  return true;
}

bool SinevAQuicksortWithSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_quicksort_with_simple_merge
