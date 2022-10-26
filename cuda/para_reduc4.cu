#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void sumReduction(int* a,int* b){ // b is the result
  __shared__ int partial_sum[256*4]; //256 elements 4 bytes per integer
  int id=blockIdx.x* blockDim.x*2+ threadIdx.x;

  //load elements into shared memory
  partial_sum[threadIdx.x]=a[id]+a[id+blockDim.x]; //at the beginning, add 01, 23, 45, two adjacent blocks together   
  //use threadIdx.x is to load element using global thread id, it automatically know dimention
  __syncthreads();




  //-------Or we can do sequential blocking!!
  for(int s=blockDim.x/2; s>0; s>>=1){ //the first half of things add with second half!
	if(threadIdx.x<s){//for each thread in a block with shared memory
		partial_sum[threadIdx.x]+=partial_sum[threadIdx.x+s];
	}
    //this only concerns one block
    __syncthreads();
  }
  
  if(threadIdx.x==0){
    b[blockIdx.x]=partial_sum[0];
  }
}

int main() {
	// Vector size
	int N = 65536;
	size_t bytes = N * sizeof(int);

	// Host data
	vector<int> h_v(N);
	vector<int> h_v_r(N);

  // Initialize the input data
  generate(begin(h_v), end(h_v), [](){ return 1; });

	// Allocate device memory
	int *d_v, *d_v_r;
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);
	
	// Copy to device
	cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);
	
	// TB Size
	const int TB_SIZE = 256;

	// Grid Size (No padding)
	int GRID_SIZE = N / TB_SIZE;

	// Call kernels
	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r); //65536 elements condense to 256 elements

	sumReduction<<<1, TB_SIZE>>> (d_v_r, d_v_r); //256 elements use a single block to 1 element

	// Copy to host;
	//cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

  cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result

	cout <<h_v_r[0];

	return 0;
}