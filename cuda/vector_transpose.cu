#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

void matrix_init(int* vec,int size,int n){
    for(int i=0;i<size;i++){
        vec[i]=n;
    }
    return;
}

__global__ void swap(int* a,int* b,int n){
    int id=(blockIdx.x * blockDim.x)+threadIdx.x; //Idx is the block index and thread id is the thread (start at 0) 
    int b_id=(n-1)-id;
    if(id<n){
        int temp=b[id];
        b[id]=a[b_id];
        a[b_id]=temp;
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

    int *a,*b; //all vectors 
    int n=10;
    size_t bytes=sizeof(int)*n;

    cudaMallocManaged(&a,bytes);
    cudaMallocManaged(&b,bytes);

    matrix_init(a,1);
    matrix_init(b,2);
    
    int thread=256;

    int block=(n+thread-1)/thread; //how many total devide how many element and round up

    vector_add<<<block,thread>>>(a,b,out,n);

    cudaDeviceSynchronize(); //this step is important so all devices are sync because no memcpy anymore

    for(int i=0;i<n;i++){
        printf("%d",a[i]);
    }
    for(int i=0;i<n;i++){
        printf("%d",b[i]);
    }
}