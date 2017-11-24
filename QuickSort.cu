/* nvcc -I/usr/local/cuda-8.0/samples/common/inc/ -arch=sm_35 -rdc=true QuickSort.cu -o QuickSort.out -lcudadevrt
*/
/*   ./QuickSort 'size' '1/0'    
	ex - QuickSort 1000 1
*/

#include <helper_cuda.h>
#include <helper_string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define MAX_DEPTH  16
#define FEW_LIMIT  32


extern "C" 
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selection_sort( int *data, int left, int right )
{
  for( int i = left ; i <= right ; ++i ){
    int min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j ){
      int val_j = data[j];
      if( val_j < min_val ){
        min_idx = j;
        min_val = val_j;
      }
    }
//printf("Selection. :");
    // Swap the values.
    if( i != min_idx ){
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}
__global__ void cuda_gpu_quicksort(int *data, int left, int right, int depth ){
    //If we're too deep or there are few elements left, we use an Selection sort...
    if( depth >= MAX_DEPTH || right-left <= FEW_LIMIT ){
        selection_sort( data, left, right );
        return;
    }

    cudaStream_t s,s1;
    int *lptr = data+left;
    int *rptr = data+right;
    int  pivot = data[(left+right)/2];

    int lval;
    int rval;
    //int count =0;
    //int size = (right -left) +1;
    //kernel_count++;
    //printf("kernel_count = %d \n",kernel_count);
    int nright, nleft;

    // Do the partitioning.
    while (lptr <= rptr){
        // Find the next left- and right-hand values to swap
        lval = *lptr;
        rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot && lptr < data+right){
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot && rptr > data+left){
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr){
            *lptr = rval;
            *rptr = lval;
            lptr++;
            rptr--;
        }
        //count++;
       // printf("iteration No. %d : \n",count);
        /*for(int i=0; i<size; i++){
        	printf("%d ",data[i]);
        }
        printf("\n");*/
    }

    // Now the recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Launch a new block to sort the left part.
    /*Creates a new asynchronous stream. The flags argument determines the behaviors of the stream. Valid values for flags are
	cudaStreamDefault: Default stream creation flag.
	cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 
	(the NULL stream), and that the created stream should perform no implicit synchronization with stream 0.
	Note:
		Note that this function may also return error codes from previous, asynchronous launches.
*/
    if (left < (rptr-data)){
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cuda_gpu_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right){
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cuda_gpu_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}


// gcc compiled code will call this function to access CUDA Quick Sort.
// This calls the kernel, which is recursive. Waits for it, then copies it's
// output back to CPU readable memory.
extern "C"
void gpu_quicksort(int *data, int n){
    int* gpuData;
    int left = 0;
    int right = n-1;
    //Setting "limit" to "value" is a request by the application to update the current limit maintained by the device. 
    // Prepare Kernel for the max depth 'MAX_DEPTH'.Current maximum synchronization depth
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Allocate GPU memory.
    cudaMalloc((void**)&gpuData,n*sizeof(int));
    cudaMemcpy(gpuData,data, n*sizeof(int), cudaMemcpyHostToDevice);

    // Launch on device
    cuda_gpu_quicksort<<< 1, 1 >>>(gpuData, left, right, 0);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(data,gpuData, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuData);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}




int main(int argc, char **argv) {
    int size=atoi(argv[1]);
    clock_t start, end;
    int i,printvector =atoi(argv[2]);
    //Initialize and declare host array of size input*sizeof(int)
    int* array;
    array = (int*)malloc(size*sizeof(int));


    srand(time(NULL));
    //vector for host //duplicate initialization
    int *vet = array;
    //storing Random Values in Array
    for(i = 0; i < size; i++) {
        array[i] = rand() % size;

    }

    int *vet_aux = (int*)malloc(sizeof(int)*size);
    // Create a copy of the vector to print it before and after it is sorted in case this 1 option is enabled
    for(i=0; i<size; i++){
        vet_aux[i] = vet[i];
    }
    // Sort the array using GPU QUICKSORT
   
            start = clock();
            gpu_quicksort(array,size);
            end = clock();

    
    if(printvector){printf("Original: ");
        for(i=0; i<size; i++){
            printf("%d ", vet_aux[i]);
        }
        printf("\n\nSorted: ");
        for(i=0; i<size; i++){
            printf("%d ", vet[i]);
        }}
        
    printf("\n-- Analysis --\n\n");
    printf("Sorting algorithm: Quick Sort\n");
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
