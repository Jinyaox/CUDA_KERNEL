#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

void matrix_init(int* vec,int size,int n){
    for(int i=0;i<size;i++){
        vec[i]=n;
    }
    return;
}

__global__ void mat_mult(int* a,int* b,int* out,int width, int p, int q){ //w is the p*width and width*q matrix so it's the "between"
    int row=(blockIdx.y * blockDim.y)+threadIdx.y; //y determines the vertical rows 
    int col=(blockIdx.x * blockDim.x)+threadIdx.x; //x determines the horizontal columns
    int temp=0; //does not matter becuase each time there will be one 
    if((row<p)&&(col<q)){
        for(int i=0;i<width;i++){
            temp+=a[row*width+i]*b[i*q+col];
        }
        out[row*q+col]=temp;
    }
}


void verify_result(vector<int> &a, vector<int> &b, vector<int> &c,int M, int N,int width) {
  // For every row...
  for (int row = 0; row < M; row++) {
    // For every column...
    for (int col = 0; col < N; col++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int i = 0; i < width; i++) {
        // Accumulate the partial results
        tmp += a[row * width + i] * b[i * N + col];
      }

      // Check against the CPU result
      assert(tmp == c[row * N + col]);
    }
  }
}


//this is the vector add in unified memory
int main() {
  // Matrix size of 1024 x 1024;
  int N = 1 << 10;

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  mat_mult<<<blocks, threads>>>(d_a, d_b, d_c, N,N,N);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c, N,N,N);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}