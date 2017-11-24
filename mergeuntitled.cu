/* nvcc -I/usr/local/cuda-8.0/samples/common/inc/ -arch=sm_35 -rdc=true QuickSort.cu -o QuickSort.out -lcudadevrt
*/
#include <helper_string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define MAX_DEPTH       16

__global__ void simple_mergesort(int* data,int *dataAux,int begin,int end, int depth){
    int middle = (end+begin)/2;
    int i0 = begin;
    int i1 = middle;
    int index;
    int n = end-begin;

    // Used to implement recursions using CUDA parallelism.
    cudaStream_t s,s1;

    if(n < 2){
        return;
    }

    //launches recursion on left and right part
    cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking);
    simple_mergesort<<< 1, 1, 0, s >>>(data,dataAux, begin, middle, depth+1);
    cudaStreamDestroy(s);
    cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking);
    simple_mergesort<<< 1, 1, 0, s1 >>>(data,dataAux, middle, end, depth+1);
    cudaStreamDestroy(s1);

 
    cudaDeviceSynchronize();


    for (index = begin; index < end; index++) {
        if (i0 < middle && (i1 >= end || data[i0] <= data[i1])){
            dataAux[index] = data[i0];
            i0++;
        }
	else{
            dataAux[index] = data[i1];
            i1++;
        }
    }

  // copy data from auxilarry memory to main memory
    for(index = begin; index < end; index++){
        data[index] = dataAux[index];
    }
}

// gcc compiled code will call this function to access CUDA Merge Sort.
extern "C"
void gpumerge_sort(int* a,int n){
    int* gpuData;
    int* gpuAuxData;
    int left = 0;
    int right = n;

   
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Allocate GPU memory.
    cudaMalloc((void**)&gpuData,n*sizeof(int));
    cudaMalloc((void**)&gpuAuxData,n*sizeof(int));
    cudaMemcpy(gpuData,a, n*sizeof(int), cudaMemcpyHostToDevice);

    // Launch on device
    simple_mergesort<<< 1, 1 >>>(gpuData,gpuAuxData, left, right, 0);
    cudaDeviceSynchronize();

    
    cudaMemcpy(a,gpuData, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuAuxData);
    cudaFree(gpuData);
    cudaDeviceReset();
}




int main(int argc, char **argv) {
    int size=atoi(argv[1]) ;
    clock_t start, end;
    int i,printvector =atoi(argv[2]);

    int* array;
    array = (int*)malloc(size*sizeof(int));


    srand(time(NULL));
    
    int *vet = array;
    for(i = 0; i < size; i++) {
        array[i] = rand() % size;

    }

    int *vet_aux = (int*)malloc(sizeof(int)*size);
    // Create a copy of the vector to print it berfore and after it is sorted in case this option is enabled
    for(i=0; i<size; i++){
        vet_aux[i] = vet[i];
    }
    // Sort the array
   
            start = clock();
            gpumerge_sort(array,size);
            end = clock();

   
       if(printvector)
        { printf("Original: ");
        for(i=0; i<size; i++){
            printf("%d ", vet_aux[i]);
        }
        printf("\n\nSorted: ");
        for(i=0; i<size; i++){
            printf("%d ", vet[i]);
        }}
     printf("\n-- Analysis --\n\n");
    printf("Sorting algorithm: MergeSort\n");
    printf("Array type:Random");
    printf("Array size: %d\n", size);
    double elapsed_time;
     elapsed_time = (((double)(end-start))/CLOCKS_PER_SEC);
    printf("Time elapsed: %f s\n", elapsed_time);
    
    free(vet);
    free(vet_aux);
    printf("\n\n");
    return 0;
}
