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

int main(){
    int *a,*b,*out; //CPU vectors
    int *g_a,*g_b,*g_out;
    int n=10;
    size_t bytes=sizeof(int)*n;
    
    //init vector on CPU
    a=(int*)malloc(bytes);
    b=(int*)malloc(bytes);
    out=(int*)malloc(bytes);

    //init vector on GPU
    cudaMalloc(&g_a,bytes);
    cudaMalloc(&g_b,bytes);
    cudaMalloc(&g_out,bytes);

    //initialize vectors with random values 0-99
    matrix_init(a,n);
    matrix_init(b,n);

    //put CPU vector to GPU
    cudaMemcpy(g_a,a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(g_b,b,bytes,cudaMemcpyHostToDevice);

    //Threadblock size
    int thread=256;

    //blocks
    int block=(n+256-1)/thread; //how many total devide how many element and round up

    vector_add<<<block,thread>>>(g_a,g_b,g_out,n);

    //calc done now copy back
    cudaMemcpy(out,g_out,bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        printf("%d",out[i]);
    }
}