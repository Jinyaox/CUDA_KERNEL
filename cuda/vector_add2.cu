#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

void matrix_init(int* vec,int size){
    for(int i=0;i<size;i++){
        vec[i]=5;
    }
    return;
}

__global__ void vector_add(int* a,int* b,int* out,int n){
    int id=(blockIdx.x * blockDim.x)+threadIdx.x; //Idx is the block index and thread id is the thread (start at 0) 
    if(id<n){
        out[id]=a[id]+b[id];
    }
}


//this is the vector add in unified memory
int main(){    
    //deleted codes//

    /*init vector on CPU
    a=(int*)malloc(bytes);
    b=(int*)malloc(bytes);
    out=(int*)malloc(bytes);

    //init vector on GPU
    cudaMalloc(&g_a,bytes);
    cudaMalloc(&g_b,bytes);
    cudaMalloc(&g_out,bytes);*/

    //instead of allocate on both GPU and CPU, we use a,b,out on Unified memory.
    //transfer will automatically happen

    //initialize vectors with random values 0-99

    /*put CPU vector to GPU
    cudaMemcpy(g_a,a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(g_b,b,bytes,cudaMemcpyHostToDevice);*/   //no need to memory copy, all these will be done by the kernel automatically

    //Valid Codes//

    int *a,*b,*out; //all vectors 
    int n=10;
    size_t bytes=sizeof(int)*n;

    cudaMallocManaged(&a,bytes);
    cudaMallocManaged(&b,bytes);
    cudaMallocManaged(&out,bytes);

    matrix_init(a,n);
    matrix_init(b,n);
    
    int thread=256;

    int block=(n+thread-1)/thread; //how many total devide how many element and round up

    vector_add<<<block,thread>>>(a,b,out,n);

    cudaDeviceSynchronize(); //this step is important so all devices are sync because no memcpy anymore

    for(int i=0;i<n;i++){
        printf("%d",out[i]);
    }
    for(int i=0;i<n;i++){
        printf("%d",a[i]+b[i]);
    }
}