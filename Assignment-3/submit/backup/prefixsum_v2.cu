#include <stdio.h>
#include <cuda.h>

#define N 64
// Kernel 1: Local scan (shared memory)
__global__ void block_prefix_sum_kernel(int *input, int *output, int *block_sums, int n)
{
    extern __shared__ int sh_data[];
    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        sh_data[tid] = input[id];
    else
        sh_data[tid] = 0;

    __syncthreads();

    // Inclusive scan in shared memory (Hillis-Steele)
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int temp = 0;
        if (tid >= offset)
            temp = sh_data[tid - offset];
        __syncthreads();
        sh_data[tid] += temp;
        __syncthreads();
    }

    if (id < n)
        output[id] = sh_data[tid];

    if (tid == blockDim.x - 1 && id < n)
        block_sums[blockIdx.x] = sh_data[tid]; // Save total sum of this block
}

// Kernel 2: Scan over block_sums (single block)
__global__ void scan_block_sums_kernel(int *block_sums, int *scanned_sums, int num_blocks)
{
    extern __shared__ int sh_data[];
    int tid = threadIdx.x;

    if (tid < num_blocks)
        sh_data[tid] = block_sums[tid];
    else
        sh_data[tid] = 0;

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int temp = 0;
        if (tid >= offset)
            temp = sh_data[tid - offset];
        __syncthreads();
        sh_data[tid] += temp;
        __syncthreads();
    }

    if (tid < num_blocks)
        scanned_sums[tid] = sh_data[tid];
}

// Kernel 3: Offset each block's output by prefix sum of previous blocks
__global__ void fix_prefix_sum_kernel(int *output, int *scanned_sums, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && id < n)
    {
        output[id] += scanned_sums[blockIdx.x - 1];
    }
}

__global__ void shift(int *pref_sum, int V)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        int tmp = 0;
        if (id > 0)
        {
            tmp = pref_sum[id - 1];
        }
        __syncthreads();
        if (id >= 0)
        {
            pref_sum[id] = tmp;
        }
    }
}

__global__ void print_prefsum(int *pref_sum, int V, int *block_sum, int block)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d ", pref_sum[i]);
    }
    printf("\n");
    for (int i = 0; i < block; i++)
    {
        printf("%d ", block_sum[i]);
    }
    printf("\n");
}

int main()
{

    // cudaSetDevice(5);
    int h_pref_sum[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int *d_pref_sum;
    cudaMalloc((void **)&d_pref_sum, sizeof(h_pref_sum));
    cudaMemcpy(d_pref_sum, h_pref_sum, sizeof(h_pref_sum), cudaMemcpyHostToDevice);
    int *d_pref_sum_out;
    cudaMalloc((void **)&d_pref_sum_out, sizeof(h_pref_sum));

    int V = 15;
    int *d_block_sum;
    int threadsPerBlock = 5;
    int block = ceil((float)V / threadsPerBlock);
    cudaMalloc((void **)&d_block_sum, sizeof(int) * block);
    cudaMemset(d_block_sum, 0, sizeof(int) * block);

    int *d_block_sum_out;
    cudaMalloc((void **)&d_block_sum_out, sizeof(int) * block);

    block_prefix_sum_kernel<<<block, threadsPerBlock, sizeof(int) * threadsPerBlock>>>(d_pref_sum, d_pref_sum, d_block_sum, V);
    scan_block_sums_kernel<<<1, block, sizeof(int) * block>>>(d_block_sum, d_block_sum, block);
    fix_prefix_sum_kernel<<<block, threadsPerBlock>>>(d_pref_sum, d_block_sum, V);

    shift<<<block, 5>>>(d_pref_sum, V);
    print_prefsum<<<1, 1>>>(d_pref_sum, V, d_block_sum, block);

    cudaThreadSynchronize();
    return 0;
}