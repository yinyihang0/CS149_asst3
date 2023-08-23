#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

#define checkCudaErrors(call)                                 \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#define checkKernelErrors()                                   \
    do {                                                      \
        cudaError_t err = cudaGetLastError();                 \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "Kernel Execution Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)



// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result


__device__ void scan_per_warp(int* share_memory, int lane, int* warp_sum)
{
    for(int two_d = 1; two_d <= 16; two_d*=2)
    {
        int two_dplus1 = 2 * two_d;
        if( (lane + 1) % two_dplus1 == 0 && threadIdx.x >= two_d)
            share_memory[threadIdx.x] += share_memory[threadIdx.x - two_d];
        
        __syncwarp(); 
    }
    if(lane == 31)
    {
        warp_sum[threadIdx.x >> 5] = share_memory[threadIdx.x];
        share_memory[threadIdx.x] = 0;
    }
    __syncwarp();
    
    for(int two_d = 16; two_d >= 1; two_d /= 2)
    {
        int two_dplus1 = 2 * two_d;
        if( (lane + 1) % two_dplus1 == 0 && threadIdx.x >= two_d)
        {
            int t = share_memory[threadIdx.x - two_d];
            share_memory[threadIdx.x - two_d] = share_memory[threadIdx.x];
            share_memory[threadIdx.x] += t;
        }
        __syncwarp();
    }
}

__device__ void scan_warp_sum(int *warp_sum, int lane, int* part_sum, int part_i)
{
    for(int two_d = 1; two_d <= 16; two_d*=2)
    {
        int two_dplus1 = 2 * two_d;
        if( (lane + 1) % two_dplus1 == 0 && threadIdx.x >= two_d)
            warp_sum[threadIdx.x] += warp_sum[threadIdx.x - two_d];
        
        __syncwarp(); 
    }
    if(lane == 31)
    {
        part_sum[part_i] = warp_sum[threadIdx.x];
        warp_sum[threadIdx.x] = 0;
    }
    __syncwarp();
    
    for(int two_d = 16; two_d >= 1; two_d /= 2)
    {
        int two_dplus1 = 2 * two_d;
        if( (lane + 1) % two_dplus1 == 0 && threadIdx.x >= two_d)
        {
            int t = warp_sum[threadIdx.x - two_d];
            warp_sum[threadIdx.x - two_d] = warp_sum[threadIdx.x];
            warp_sum[threadIdx.x] += t;
        }
        __syncwarp();
    }
}

__device__ void scan_per_block(int* share_memory, int* part_sum, int part_i)
{
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    // make sure blockDim.x / 32 == 32
    __shared__ int warp_sum[32];
    scan_per_warp(share_memory, lane, warp_sum);
    __syncthreads();

    // if(lane == 31)
    //     warp_sum[warp_id] = share_memory[threadIdx.x];
    // __syncthreads();

    // scan warp sum and set part_sum
    if(warp_id == 0)
    {
        scan_warp_sum(warp_sum, lane, part_sum, part_i);
    }
    __syncthreads();

    // add warp sum
    if(warp_id > 0)
        share_memory[(warp_id << 5) + lane] += warp_sum[warp_id];
    __syncthreads();

}

__global__ void exclusive_scan_per_block(int* input, int *result, int* part_sum, int N, int part_num)
{
    // __shared__ int share_memory[blockDim.x];
    __shared__ int share_memory[1024];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        size_t thread_id = part_i * blockDim.x + threadIdx.x;
        if(thread_id < N)
            share_memory[threadIdx.x] = input[thread_id];

        __syncthreads();
        scan_per_block(share_memory, part_sum, part_i);
        __syncthreads();

        if(thread_id < N)
            result[thread_id] = share_memory[threadIdx.x];
        // if(threadIdx.x == blockDim.x - 1)
        //     part_sum[part_i] = share_memory[threadIdx.x];
    }
} 

__global__ void scan_part_sum(int* part_sum, int part_num)
{
    // __shared__ int share_memory[part_num];
    // part_sum[0] = 0;
    for(int i = 1; i < part_num; ++i)
    {
        part_sum[i] += part_sum[i-1];
    }

}

__global__ void add_part_sum(int* result, int* part_sum, int part_num, int N)
{
    // 有partnum个base 需要加到里面，每个base需要加blocksize次数
    // __shared__ int* share_memory[blockDim.x];
    __shared__ int share_memory[1024];
    for(size_t i = blockIdx.x; i < part_num; i+=gridDim.x)
    {
        int thread_id = threadIdx.x + i * blockDim.x;
        if(thread_id < N)
            share_memory[threadIdx.x] = result[thread_id];
        __syncthreads();

        // __shared__ int part_sum_i = part_sum[i];
        if(i > 0)
            share_memory[threadIdx.x] += part_sum[i-1];
        __syncthreads();

        if(thread_id < N)
            result[thread_id] = share_memory[threadIdx.x];
        __syncthreads();
    }
}

void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    size_t block_size = 1024;
    size_t part_num = (N + block_size - 1) / block_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    int* part_sum = nullptr;

    cudaMalloc((void **)&part_sum, sizeof(int)*part_num);

    exclusive_scan_per_block<<<block_num, block_size>>>(input, result, part_sum, N, part_num);
    cudaDeviceSynchronize();
    checkKernelErrors();
    scan_part_sum<<<1, 1>>>(part_sum, part_num);
    cudaDeviceSynchronize();
    checkKernelErrors();
    add_part_sum<<<block_num, block_size>>>(result, part_sum, part_num, N);
    cudaDeviceSynchronize();
    checkKernelErrors();
    cudaFree(part_sum);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void set_flag(int* device_input, int length, int part_num, int* temp)
{
    __shared__ int share_memory[1024];
    __shared__ int share_memory_shift[1024];
    for(size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x)
    {
        size_t thread_id = threadIdx.x + part_i * blockDim.x;
        if(thread_id < length)
        {
            share_memory[threadIdx.x] = device_input[thread_id];
            if(threadIdx.x < length - 1)
                share_memory_shift[threadIdx.x] = device_input[thread_id + 1];

            __syncthreads();
            if(share_memory[threadIdx.x] == share_memory_shift[threadIdx.x])
            {
                temp[thread_id] = 1;
            }
            else{
                temp[thread_id] = 0;
            }
        }
        __syncthreads();
    }
}
__global__ void scatter(int* temp, int* position, int* device_output, int part_num ,int length)
{
    // __shared__ int device_input_shm[1024];
    __shared__ int temp_shm[1024];
    __shared__ int position_shm[1024];

    for(size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x)
    {
        size_t thread_id = threadIdx.x + part_i * blockDim.x;
        if(thread_id < length)
        {
            // device_input_shm[threadIdx.x] = device_input[thread_id];
            temp_shm[threadIdx.x] = temp[thread_id];
            position_shm[threadIdx.x] = position[thread_id];
            __syncthreads();

            if(temp_shm[threadIdx.x] == 1)
            {
                device_output[position_shm[threadIdx.x]] = thread_id;
            }
        }
        __syncthreads();
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int* temp = nullptr;
    int* scan_temp = nullptr;
    int* length_ = new int[1];

    checkCudaErrors(cudaMalloc((void**)&temp, sizeof(int) * length));
    checkCudaErrors(cudaMalloc((void**)&scan_temp, sizeof(int) * length));

    checkKernelErrors();

    int block_size = 1024;
    int part_num = (length + block_size - 1) / block_size;
    int block_num = std::min(128, part_num);

    set_flag<<<block_num, block_size>>>(device_input, length, part_num, temp);
    cudaDeviceSynchronize();
    checkKernelErrors();

    exclusive_scan(temp, length, scan_temp);
    cudaDeviceSynchronize();
    checkKernelErrors();

    scatter<<<block_num, block_size>>>(temp, scan_temp, device_output, part_num, length);
    cudaDeviceSynchronize();
    checkKernelErrors();

    checkCudaErrors(cudaMemcpy(length_, scan_temp+length-1, sizeof(int), cudaMemcpyDeviceToHost));
    
    checkKernelErrors();

    cudaFree(temp);
    cudaFree(scan_temp);
    return *length_; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    std::cout << "start_find_repeat" << std::endl;
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
