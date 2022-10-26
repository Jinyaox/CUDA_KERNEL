// This program implements a 1D convolution using CUDA
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      n       = number of elements in array
//      m       = number of elements in the mask 
__global__ void convolution_1d(int *array, int *mask, int *result, int n,int m) {
    //each value in the result will be calculated using one thread!
    int tid=blockIdx.x*blockDim.x+threadIdx.x; //so global thread ID->position calculation 

    int r=m/2; //the radius (half) of the mask so there are negative index at the beginning/edge

    int start=tid-r; //first element where to start, may be -1 when there is edge case

    //------------------------------
    //Edge Case:
    //   1,2,3,4,5,6,7 [inputs]
    // 1,5,7           [mask and starting at -1]
    //-------------------------------

    int temp=0;

    //each value in the result will be calculated using one thread!
    for(int j=0;j<m;j++){
        if(((start+j)>=0) && ((start+j)<n)){
            temp+=array[start+j]*mask[j];
        }
    }

    result[tid]=temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n, int m) {
  int radius = m / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}

int main() {
  // Number of elements in result array
  int n = 1 << 20;

  // Size of the array in bytes
  int bytes_n = n * sizeof(int);

  // Number of elements in the convolution mask
  int m = 7;

  // Size of mask in bytes
  int bytes_m = m * sizeof(int);

  // Allocate the array (include edge elements)...
  std::vector<int> h_array(n);

  // ... and initialize it
  std::generate(begin(h_array), end(h_array), [](){ return rand() % 100; });

  // Allocate the mask and initialize it
  std::vector<int> h_mask(m);
  std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

  // Allocate space for the result
  std::vector<int> h_result(n);

  // Allocate space on the device
  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  // Copy the data to the device
  cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

  // Threads per TB
  int THREADS = 256;

  // Number of TBs
  int GRID = (n + THREADS - 1) / THREADS;

  // Call the kernel
  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  // Copy back the result
  cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array.data(), h_mask.data(), h_result.data(), n, m);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory on the device and host
  cudaFree(d_result);
  cudaFree(d_mask);
  cudaFree(d_array);

  return 0;
}