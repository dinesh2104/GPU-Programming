#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void init_kernel(long int *d_mat, int new_h, int new_w, int c)
{
    int i = blockIdx.y;
    int j = threadIdx.x;
    int k = blockIdx.x;
    if (i < new_h && j < new_w && k < c)
    {
        d_mat[i * blockDim.x + j + k * gridDim.y * blockDim.x] = 0;
    }
}

__global__ void copy_kernel(long int *d_mat, long int *mat, int h, int w, int c, int padding_h, int padding_w, int new_h, int new_w)
{
    int i = blockIdx.y;
    int j = threadIdx.x;
    int k = blockIdx.x;
    if (i < h && j < w && k < c)
    {
        d_mat[(k * new_h * new_w) + ((i + padding_h) * new_w) + j + padding_w] = mat[i * blockDim.x + j + k * gridDim.y * blockDim.x];
    }
}

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{
    extern __shared__ long int shared_mem[];

    long int *shared_matrix = shared_mem;
    long int *shared_filter = &shared_mem[r * new_w * c];

    int ii = blockIdx.y;
    int jj = threadIdx.x;
    int kk = blockIdx.x;
    if (threadIdx.x == 0)
    {
        if (ii < h && jj < w && kk < k)
        {
            for (int p = 0; p < r && (ii + p) < new_h; p++)
            {
                for (int q = 0; q < new_w; q++)
                {
                    for (int z = 0; z < c; z++)
                    {

                        shared_matrix[(z * r * new_w) + (p * new_w) + q] = matrix[(z * new_h * new_w) + ((ii + p) * new_w) + q];
                    }
                }
            }
            for (int p = 0; p < r; p++)
            {
                for (int q = 0; q < s; q++)
                {
                    for (int z = 0; z < c; z++)
                    {
                        shared_filter[(z * r * s) + (p * s) + q] = filter[(kk * c * r * s) + (z * r * s) + (p * s) + q];
                    }
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("The matrix\n");
        for (int z = 0; z < c; z++)
        {
            for (int p = 0; p < r; p++)
            {
                for (int q = 0; q < new_w; q++)
                {

                    printf("%d ", shared_matrix[(z * r * new_w) + (p * new_w) + q]);
                }
                printf("\n");
            }
        }
    }

    // Computations

    if (ii < h && jj < w && kk < k)
    {
        long int sum = 0;
        for (int x = 0; x < r; x++)
        {
            for (int y = 0; y < s; y++)
            {
                for (int z = 0; z < c; z++)
                {
                    sum += shared_matrix[(z * r * new_w) + (x * new_w) + jj + y] * shared_filter[(kk * c * r * s) + (z * r * s) + (x * s) + y];
                }
            }
        }
        result[(kk * h * w) + (ii * w) + jj] = sum;
    }
}

__constant__ long int shared_filter[4096];

__global__ void dkernelv1(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{
    int i = blockIdx.y;
    int j = threadIdx.x;
    int l = blockIdx.x;
    if (i < h && j < w && l < k)
    {
        long int sum = 0;
        for (int z = 0; z < c; z++)
        {
            for (int x = 0; x < r; x++)
            {
                for (int y = 0; y < s; y++)
                {

                    sum += matrix[(z * new_h * new_w) + ((i + x) * new_w) + j + y] * shared_filter[(l * c * r * s) + (z * r * s) + (x * s) + y];
                }
            }
        }
        result[(l * h * w) + (i * w) + j] = sum;
    }
}

// This approach is taking more time.
__global__ void dkernelv2(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{

    int ii = blockIdx.y;
    int jj = blockIdx.z;
    int kk = blockIdx.x;

    __shared__ long int shared_mem[4096];

    long int *shared_matrix = shared_mem;

    if (ii < h && jj < w && kk < k)
    {
        for (int p = 0; p < r; p++)
        {
            for (int q = 0; q < s; q++)
            {
                for (int z = 0; z < c; z++)
                {

                    shared_matrix[(z * r * s) + (p * s) + q] = matrix[(z * new_h * new_w) + ((ii + p) * new_w) + jj + q];
                }
            }
        }
    }

    long int sum = 0;

    if (ii < h && jj < w && kk < k)
    {
        for (int z = 0; z < c; z++)
        {
            for (int p = 0; p < r; p++)
            {
                for (int q = 0; q < s; q++)
                {
                    sum += shared_matrix[(z * r * s) + (p * s) + q] * shared_filter[(kk * c * r * s) + (z * r * s) + (p * s) + q];
                }
            }
        }
    }
    result[(kk * h * w) + (ii * w) + jj] = sum;
}

// This approach is giving good time
__global__ void dkernelv3(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{
    int i = blockIdx.y;
    int j = threadIdx.x;
    int l = blockIdx.x;

    __shared__ long int shared_filter1[4096];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < r * s * c * k; i++)
        {
            shared_filter1[i] = filter[i];
        }
    }
    __syncthreads();

    if (i < h && j < w && l < k)
    {
        long int sum = 0;
        for (int z = 0; z < c; z++)
        {
            for (int x = 0; x < r; x++)
            {
                for (int y = 0; y < s; y++)
                {

                    sum += matrix[(z * new_h * new_w) + ((i + x) * new_w) + j + y] * shared_filter1[(l * c * r * s) + (z * r * s) + (x * s) + y];
                }
            }
        }
        result[(l * h * w) + (i * w) + j] = sum;
    }
}

// This approach work but if the r value is more than 32 then it will not work.
__global__ void dkernelv4(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{
    int ii = blockIdx.y;
    int jj = blockIdx.z * blockDim.x + threadIdx.x;
    int kk = blockIdx.x;

    __shared__ long int shared_mem[6144];

    long int *shared_matrix = shared_mem;

    if (threadIdx.x == 0)
    {
        if (ii < h && jj < w && kk < k)
        {
            for (int z = 0; z < c; z++)
            {
                for (int p = 0; p < r; p++)
                {
                    for (int q = 0; q < s + 32 && q < new_w; q++)
                    {

                        shared_matrix[(z * r * (s + 32)) + (p * (s + 32)) + q] = matrix[(z * new_h * new_w) + ((ii + p) * new_w) + jj + q];
                    }
                }
            }
        }
    }
    __syncthreads();

    long int sum = 0;
    if (ii < h && jj < w && kk < k)
    {
        for (int z = 0; z < c; z++)
        {
            for (int p = 0; p < r; p++)
            {
                for (int q = 0; q < s; q++)
                {
                    sum += shared_matrix[(z * r * (s + 32)) + (p * (s + 32)) + q + threadIdx.x] * shared_filter[(kk * c * r * s) + (z * r * s) + (p * s) + q];
                }
            }
        }
        result[(kk * h * w) + (ii * w) + jj] = sum;
    }
}

__global__ void dkernelv5(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k, int new_h, int new_w)
{
    int ii = blockIdx.y;
    int jj = blockIdx.z;
    int kk = blockIdx.x;
    int pp = threadIdx.x;
    int qq = threadIdx.y;

    __shared__ long int shared_mem[4096];
    long int *shared_matrix = shared_mem;

    if (ii < h && jj < w && kk < k)
    {
        for (int z = 0; z < c; z++)
        {

            shared_matrix[(z * r * s) + (pp * s) + qq] = matrix[(z * new_h * new_w) + ((ii + pp) * new_w) + jj + qq];
        }
    }

    __syncthreads();

    long int sum = 0;

    if (ii < h && jj < w && kk < k && pp < r)
    {
        for (int z = 0; z < c; z++)
        {

            sum += shared_matrix[(z * r * s) + (pp * s) + qq] * shared_filter[(kk * c * r * s) + (z * r * s) + (pp * s) + qq];
        }
        atomicAdd((unsigned long long int *)&result[(kk * h * w) + (ii * w) + jj], (unsigned long long int)sum);
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    int padding_h = (r - 1) / 2;
    int padding_w = (s - 1) / 2;
    int new_h = 2 * padding_h + h;
    int new_w = 2 * padding_w + w;

    //  creating variable to hold memory on device(GPU)
    long int *d_mat, *mat, *d_filter, *d_ans;

    long int *prt_mat = new long int[c * new_h * new_w];

    cudaMemcpyToSymbol(shared_filter, h_filter, r * s * c * k * sizeof(long int));

    cudaMalloc(&mat, c * h * w * sizeof(long int));
    cudaMemcpy(mat, h_mat, c * h * w * sizeof(long int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_mat, c * new_h * new_w * sizeof(long int));
    // init_kernel<<<dim3(c, new_h, 1), new_w>>>(d_mat, new_h, new_w, c);

    // Kernel to copy the image data to the padded matrix.
    copy_kernel<<<dim3(c, h), w>>>(d_mat, mat, h, w, c, padding_h, padding_w, new_h, new_w);

    // cudaMemcpy(prt_mat, d_mat, c * new_h * new_w * sizeof(long int), cudaMemcpyDeviceToHost);
    // for (int k1 = 0; k1 < c; k1++)
    // {
    //     for (int i = 0; i < new_h; i++)
    //     {
    //         for (int j = 0; j < new_w; j++)
    //         {
    //             cout << prt_mat[k1 * new_h * new_w + i * new_w + j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }

    // copying the filter to the device
    cudaMalloc(&d_filter, r * s * c * k * sizeof(long int));
    cudaMemcpy(d_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    // Memory allocation for the final answer
    cudaMalloc(&d_ans, h * w * k * sizeof(long int));

    // Kernel to perform the convolution operation

    // dkernel<<<dim3(k, h), w>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    // Better approach with 0.2 but no shared mem
    //  dkernelv1<<<dim3(k, h), w>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    // dkernelv2<<<dim3(k, h, w), 1>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);
    //  gpuErrchk(cudaPeekAtLastError());

    // Good timing approach
    dkernelv3<<<dim3(k, h), w>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    // int bz = ceil((float)w / 32);
    // dkernelv4<<<dim3(k, h, bz), 32>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    // dkernelv5<<<dim3(k, h, w), dim3(r, s)>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    gpuErrchk(cudaPeekAtLastError());

    // int bz = ceil((float)w / s);
    // printf("%d - %d - %d - %d\n", h, w, bz, s);
    // dkernel<<<dim3(k, h, bz), s>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    gpuErrchk(cudaMemcpy(h_ans, d_ans, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost));
    // for (int k1 = 0; k1 < k; k1++)
    // {
    //     for (int i = 0; i < h; i++)
    //     {
    //         for (int j = 0; j < w; j++)
    //         {
    //             cout << h_ans[k1 * h * w + i * w + j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }
    cudaFree(d_mat);
    cudaFree(d_filter);
    cudaFree(d_ans);
    cudaFree(mat);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
