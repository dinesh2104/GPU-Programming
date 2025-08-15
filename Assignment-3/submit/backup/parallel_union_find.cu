#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define N 1024 // Max number of nodes

__device__ int d_parent[N];
__device__ int d_rank[N];

__device__ int find_root(int x)
{
    while (x != d_parent[x])
    {
        // Path halving
        int parent = d_parent[x];
        int grandparent = d_parent[parent];
        atomicExch(&d_parent[x], grandparent);
        x = grandparent;
    }
    return x;
}

__device__ void union_sets(int x, int y)
{
    while (true)
    {
        int rootX = find_root(x);
        int rootY = find_root(y);
        if (rootX == rootY)
            return;

        if (d_rank[rootX] < d_rank[rootY])
        {
            if (atomicCAS(&d_parent[rootX], rootX, rootY) == rootX)
                return;
        }
        else
        {
            if (atomicCAS(&d_parent[rootY], rootY, rootX) == rootY)
            {
                if (d_rank[rootX] == d_rank[rootY])
                    atomicAdd(&d_rank[rootX], 1);
                return;
            }
        }
    }
}

__global__ void init_sets(int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        d_parent[idx] = idx;
        d_rank[idx] = 0;
    }
}

__global__ void parallel_unions(int *u, int *v, int E)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < E)
    {
        union_sets(u[idx], v[idx]);
    }
}

__global__ void print_roots(int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V)
    {
        int root = find_root(idx);
        printf("Node %d -> Root %d\n", idx, root);
    }
}

// Host code
int main()
{
    const int V = 6;
    const int E = 4;
    int h_u[E] = {0, 1, 2, 4};
    int h_v[E] = {1, 2, 3, 5};

    int *d_u, *d_v;
    cudaMalloc(&d_u, E * sizeof(int));
    cudaMalloc(&d_v, E * sizeof(int));
    cudaMemcpy(d_u, h_u, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, E * sizeof(int), cudaMemcpyHostToDevice);

    init_sets<<<1, V>>>(V);
    cudaDeviceSynchronize();

    parallel_unions<<<1, E>>>(d_u, d_v, E);
    cudaDeviceSynchronize();

    print_roots<<<1, V>>>(V);
    cudaDeviceSynchronize();

    cudaFree(d_u);
    cudaFree(d_v);
    return 0;
}
