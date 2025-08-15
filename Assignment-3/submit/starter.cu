#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#define MOD 1000000007

using namespace std;
using std::cin;
using std::cout;
struct Edge
{
    int src, dest, weight, type;
    Edge() {}
};

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

__global__ void init_longlong_array(long long int *arr, long long int value, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        arr[idx] = value;
    }
}

__global__ void print(long long int *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%lld ", arr[i]);
    }
    printf("\n");
}

__global__ void clearminidx(long long int *arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        arr[idx] = LLONG_MAX;
    }
}

// Kernel code
__global__ void init(int *find, int V)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        find[id] = id;
    }
}

__global__ void printMst(Edge *mstedge, int V, int *find, long int *d_ans)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d %d %d\n", mstedge[i].src, mstedge[i].dest, mstedge[i].weight);
    }
    for (int i = 0; i < V; i++)
    {
        printf("%d ", find[i]);
    }
    printf("Final Answer: %ld\n", *d_ans);
}

__device__ int find_root(int x, int *d_parent)
{
    // Path compression

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

__device__ void union_sets(int x, int y, int *d_parent, int *d_rank)
{
    while (true)
    {
        int rootX = find_root(x, d_parent);
        int rootY = find_root(y, d_parent);
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

// TODO: There is the Race condition in adding the edge to the list fix it.
__global__ void find_minedge(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int V, Edge *mstedge, int *find, int *end_flag, long long int *minedgeidx)
{
    int v_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_id < V)
    {
        int parent = find_root(v_id, find);
        for (int i = startidx[v_id]; i < startidx[v_id] + outdegree[v_id]; i++)
        {
            int dest = dest_arr[i];
            int weight = weight_arr[i];
            // create a long int varibale with first half is weight and rest is the i value.
            long long int edge = ((long long int)weight << 32) | (long long int)i;
            if (find_root(dest, find) != parent)
            {
                // printf("Edge: %lld %d %d %lld %lld\n", edge, i, weight, edge & 0xFFFFFFFF, minedgeidx[parent]);

                atomicMin((unsigned long long int *)&minedgeidx[parent], edge);
                *end_flag = 1;
            }
        }
    }
}

__global__ void construct_minedge(Edge *mstedge, long long int *minedgeidx, int V, int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int *src_arr)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        long long int edge = minedgeidx[id];
        if (edge != LLONG_MAX)
        {
            int e_id = edge & 0xFFFFFFFF;

            mstedge[id].src = src_arr[e_id];
            mstedge[id].dest = dest_arr[e_id];
            mstedge[id].weight = weight_arr[e_id];
        }
    }
}

__global__ void removeEdge(int V, Edge *mstedge, long int *d_ans)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= V)
        return;
    for (int i = 0; i < V; i++)
    {
        if (mstedge[i].src == mstedge[id].dest && mstedge[i].dest == mstedge[id].src && mstedge[id].src > mstedge[id].dest)
        {
            mstedge[i].src = -1;
            mstedge[i].dest = -1;
            mstedge[i].weight = 0;
        }
    }
    if (mstedge[id].src != -1 && mstedge[id].dest != -1)
    {
        atomicAdd((long long unsigned *)d_ans, mstedge[id].weight);
        *d_ans = *d_ans % MOD;
    }
}

__global__ void markComponent(int V, Edge *mstedge, int *find, int *flag, int *rank)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        int src = mstedge[id].src;
        int dest = mstedge[id].dest;
        int parent_src = find_root(src, find);
        int parent_dest = find_root(dest, find);
        if (src != -1 && dest != -1 && parent_src != parent_dest)
        {
            union_sets(src, dest, find, rank);
            *flag = 1;
        }
    }
}

__global__ void clear(Edge *mstedge, int V)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        mstedge[id].src = -1;
        mstedge[id].dest = -1;
        mstedge[id].weight = 0;
    }
}

// Kernel for building CSR representation
__global__ void build_CSR(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int *src_arr, Edge *edges, int E)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < E)
    {
        int src = edges[id].src;
        int dest = edges[id].dest;
        int weight = edges[id].weight * edges[id].type;

        // Add the edge to the CSR representation
        int idx = atomicAdd(&startidx[src], 1);
        src_arr[idx] = src;
        dest_arr[idx] = dest;
        weight_arr[idx] = weight;

        int idx1 = atomicAdd(&startidx[dest], 1);
        src_arr[idx1] = dest;
        dest_arr[idx1] = src;
        weight_arr[idx1] = weight;
    }
}
__global__ void copy(int *src, int *dest, int V)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        dest[id] = src[id];
    }
}

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

__global__ void fix_prefix_sum_kernel(int *output, int *scanned_sums, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && id < n)
    {
        output[id] += scanned_sums[blockIdx.x - 1];
    }
}

__global__ void shift(int *pref_sum, int V, int *pref_sum1)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        if (id > 0)
        {
            pref_sum1[id] = pref_sum[id - 1];
        }
    }
}

__global__ void printEdge(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int V)
{
    for (int id = 0; id < V; id++)
    {

        for (int i = startidx[id]; i < startidx[id] + outdegree[id]; i++)
        {
            printf("%d %d %d\n", id, dest_arr[i], weight_arr[i]);
        }
    }
}

int main()
{
    int V;
    cin >> V;
    int E;
    cin >> E;
    Edge *graph = (Edge *)malloc(E * sizeof(Edge));

    Edge *edges = (Edge *)malloc(E * sizeof(Edge));
    int i = 0;

    int *outdegree = (int *)malloc(V * sizeof(int));
    for (int j = 0; j < V; j++)
    {
        outdegree[j] = 0;
    }

    while (i < E)
    {
        int u, v, wt, type;
        string s;
        cin >> u >> v >> wt;
        cin >> s;
        // Adding the edge to the vector.

        if (s == "green")
        {
            type = 2;
        }
        else if (s == "traffic")
        {
            type = 5;
        }
        else if (s == "dept")
        {
            type = 3;
        }
        else
        {
            type = 1;
        }
        graph[i].src = u;
        graph[i].dest = v;
        graph[i].weight = wt;
        graph[i].type = type;
        outdegree[u]++;
        outdegree[v]++;
        i++;
    }

    // Variable declaration
    int block = ceil((float)V / 1024);
    Edge *d_graph;
    int *d_startidx, *block_sum, *d_startidx1, *d_outdegree, *d_dest_arr, *d_weight_arr;
    int edge_block = ceil((float)E / 1024);
    long int *d_ans;
    long int *h_ans = (long int *)malloc(sizeof(long int));
    int *flag;
    int *d_rank;
    int *d_src_arr;

    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();

    // Kernel call(s) here
    cudaMalloc(&d_rank, V * sizeof(int));
    cudaMemset(d_rank, 0, V * sizeof(int));

    cudaMalloc(&d_src_arr, V * sizeof(int));

    cudaHostAlloc(&flag, sizeof(int), 0);
    cudaMalloc(&d_graph, E * sizeof(Edge));
    cudaMemcpy(d_graph, graph, E * sizeof(Edge), cudaMemcpyHostToDevice);

    cudaMalloc(&d_startidx, V * sizeof(int));
    cudaMemcpy(d_startidx, outdegree, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&block_sum, block * sizeof(int));
    block_prefix_sum_kernel<<<block, 1024, sizeof(int) * 1024>>>(d_startidx, d_startidx, block_sum, V);
    scan_block_sums_kernel<<<1, block, sizeof(int) * block>>>(block_sum, block_sum, block);
    fix_prefix_sum_kernel<<<block, 1024>>>(d_startidx, block_sum, V);
    cudaMalloc(&d_startidx1, V * sizeof(int));
    shift<<<block, 1024>>>(d_startidx, V, d_startidx1);

    cudaMalloc(&d_outdegree, V * sizeof(int));
    cudaMemcpy(d_outdegree, outdegree, V * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_dest_arr, 2 * E * sizeof(int));
    cudaMalloc(&d_weight_arr, 2 * E * sizeof(int));

    // print<<<1, 1>>>(d_startidx1, V);
    // cudaDeviceSynchronize();
    // cout << endl;
    build_CSR<<<edge_block, 1024>>>(d_startidx1, d_outdegree, d_dest_arr, d_weight_arr, d_src_arr, d_graph, E);
    cudaMemset(d_startidx1, 0, V * sizeof(int));
    shift<<<block, 1024>>>(d_startidx, V, d_startidx1);
    copy<<<block, 1024>>>(d_startidx1, d_startidx, V);

    // print<<<1, 1>>>(d_outdegree, V);
    // print<<<1, 1>>>(d_startidx, V);

    // printEdge<<<1, 1>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V);
    //    Result of  MST

    cudaMalloc(&d_ans, sizeof(long int));
    cudaMemset(d_ans, 0, sizeof(long int));

    Edge *d_mstEdge;
    cudaMalloc(&d_mstEdge, V * sizeof(Edge));

    // Step1: Initialize the find array
    int *d_find_arr;
    cudaMalloc(&d_find_arr, V * sizeof(int));
    init<<<block, 1024>>>(d_find_arr, V);

    int *end_flag;
    cudaHostAlloc(&end_flag, sizeof(int), 0);

    long long int *d_minedgeidx;
    cudaMalloc(&d_minedgeidx, V * sizeof(long long int));
    init_longlong_array<<<block, 1024>>>(d_minedgeidx, LLONG_MAX, V);

    // print<<<1, 1>>>(d_minedgeidx, V);

    for (int i = 0; i < 2; i++)
    {
        // Step2: Find the Minimum Edge.
        *end_flag = 0;
        find_minedge<<<block, 1024>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V, d_mstEdge, d_find_arr, end_flag, d_minedgeidx);
        // print<<<1, 1>>>(d_minedgeidx, V);
        construct_minedge<<<block, 1024>>>(d_mstEdge, d_minedgeidx, V, d_startidx, d_outdegree, d_dest_arr, d_weight_arr, d_src_arr);

        // printMst<<<1, 1>>>(d_mstEdge, V, d_find_arr, d_ans);

        cudaDeviceSynchronize();
        // if (*end_flag == 0)
        // {
        //     break;
        // }

        // Step3: Removing the mirror edges.
        removeEdge<<<block, 1024>>>(V, d_mstEdge, d_ans);

        // Step4: Marking the connected components
        while (true)
        {
            *flag = 0;
            markComponent<<<block, 1024>>>(V, d_mstEdge, d_find_arr, flag, d_rank);
            cudaDeviceSynchronize();
            if (*flag == 0)
            {
                break;
            }
        }
        //  Step5: Clearing the MSTList array
        // printMst<<<1, 1>>>(d_mstEdge, V, d_find_arr, d_ans);
        cudaDeviceSynchronize();
        clearminidx<<<block, 1024>>>(d_minedgeidx, V);
        clear<<<block, 1024>>>(d_mstEdge, V);
    }

    // Final memcpy
    cudaMemcpy(h_ans, d_ans, sizeof(long int), cudaMemcpyDeviceToHost);
    cout << "Final Answer: " << *h_ans << endl;
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // Print only the total MST weight

    // cout << elapsed1.count() << " s\n";

    cudaFree(d_startidx);
    cudaFree(d_outdegree);
    cudaFree(d_dest_arr);
    cudaFree(d_weight_arr);
    cudaFree(d_ans);

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        file << *h_ans << " ";

        file << "\n";

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