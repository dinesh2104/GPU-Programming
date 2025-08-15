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

    cudaMalloc(&mat, c * h * w * sizeof(long int));
    cudaMemcpy(mat, h_mat, c * h * w * sizeof(long int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_mat, c * new_h * new_w * sizeof(long int));
    // init_kernel<<<dim3(c, new_h, 1), new_w>>>(d_mat, new_h, new_w, c);

    // Kernel to copy the image data to the padded matrix.
    copy_kernel<<<dim3(c, h), w>>>(d_mat, mat, h, w, c, padding_h, padding_w, new_h, new_w);

    // copying the filter to the device
    cudaMalloc(&d_filter, r * s * c * k * sizeof(long int));
    cudaMemcpy(d_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    // Memory allocation for the final answer
    cudaMalloc(&d_ans, h * w * k * sizeof(long int));

    // Kernel to perform the convolution operation
    dkernel<<<dim3(k, h), w>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k, new_h, new_w);

    cudaMemcpy(h_ans, d_ans, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost);

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

    cudaDeviceSynchronize();
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
