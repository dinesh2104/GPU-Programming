#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>

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

__global__ void print_kernel(Edge *edges, int V, int E)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E)
    {
        printf("%d %d %d\n", edges[i].src, edges[i].dest, edges[i].weight);
    }
}

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

__global__ void printAnswer(long int *d_ans)
{
    printf("Final Answer: %ld\n", *d_ans);
}

__global__ void print_prefsum(int V, int *prefix_sum)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d ", prefix_sum[i]);
    }
}

__global__ void print(int *arr, int V)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d ", arr[i]);
    }
}

//----------------------------------------------------------------------------
__global__ void build_CSR(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, Edge *edges, int E)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < E)
    {
        int src = edges[id].src;
        int dest = edges[id].dest;
        int weight = edges[id].weight * edges[id].type;

        // Add the edge to the CSR representation
        int idx = atomicAdd(&startidx[src], 1);
        dest_arr[idx] = dest;
        weight_arr[idx] = weight;

        int idx1 = atomicAdd(&startidx[dest], 1);
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

__global__ void find_minedge(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int V, Edge *mstedge)
{
    int v_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (v_id < V)
    {
        int min_weight = INT_MAX;
        int min_index = -1;
        for (int i = startidx[v_id]; i < startidx[v_id] + outdegree[v_id]; i++)
        {
            if (weight_arr[i] < min_weight)
            {
                min_weight = weight_arr[i];
                min_index = i;
            }
        }
        if (min_index != -1)
        {
            mstedge[v_id].src = v_id;
            mstedge[v_id].dest = dest_arr[min_index];
            mstedge[v_id].weight = min_weight;
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
        if (mstedge[i].src == mstedge[id].dest && mstedge[i].dest == mstedge[id].src && mstedge[id].src < mstedge[id].dest)
        {
            mstedge[id].src = -1;
            mstedge[id].dest = -1;
            mstedge[id].weight = 0;
            break;
        }
    }
    if (mstedge[id].src != -1 && mstedge[id].dest != -1)
    {
        atomicAdd((long long unsigned *)d_ans, mstedge[id].weight);
        //*d_ans = *d_ans % MOD;
    }
}

__global__ void markInitComponent(int V, Edge *mstedge, int *find)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        if (mstedge[id].src == -1 && mstedge[id].dest == -1)
        {
            find[id] = id;
        }
        else
        {
            find[id] = mstedge[id].dest;
        }
    }
}

__global__ void markComponent(int V, Edge *mstedge, int *find, int *flag)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < V)
    {
        if (mstedge[id].src != -1 && mstedge[id].dest != -1)
        {
            int old_val = find[id];
            int new_val = find[old_val];
            if (old_val != new_val)
            {
                find[id] = new_val;
                *flag = 1;
            }
        }
    }
}

__global__ void mark_newvertice(int V, int *find, int *prefix_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < V)
    {
        prefix_sum[find[id]] = 1;
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

__global__ void count_outdegree(int *startidx, int *outdegree, int *dest_arr, int V, int *new_outdegree, int *find, int *pref_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        for (int i = startidx[id]; i < startidx[id] + outdegree[id]; i++)
        {
            int dest = dest_arr[i];
            if (find[id] != find[dest])
            {
                int loc = pref_sum[find[id]];
                atomicAdd(&new_outdegree[loc], 1);
            }
        }
    }
}

__global__ void find_startIndex(int *outdegree, int V, int *startidx)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        startidx[id] = outdegree[id];
    }
}

__global__ void addEdges(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int V, int *pref_sum, int *new_startidx, int *new_dest_arr, int *new_weigth_arr, int *find)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        for (int i = startidx[id]; i < startidx[id] + outdegree[id]; i++)
        {
            int dest = dest_arr[i];
            if (find[id] != find[dest])
            {

                int loc = pref_sum[find[id]];
                // printf("Adding edge %d %d %d %d\n", id, dest, loc, pref_sum[find[dest]]);
                int idx = atomicAdd(&new_startidx[loc], 1);
                new_dest_arr[idx] = pref_sum[find[dest]];
                new_weigth_arr[idx] = weight_arr[i];
            }
        }
    }
}

int main()
{
    // Create a sample graph
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
    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();

    // Kernel call(s) here

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
    build_CSR<<<edge_block, 1024>>>(d_startidx1, d_outdegree, d_dest_arr, d_weight_arr, d_graph, E);
    cudaMemset(d_startidx1, 0, V * sizeof(int));
    shift<<<block, 1024>>>(d_startidx, V, d_startidx1);
    copy<<<block, 1024>>>(d_startidx1, d_startidx, V);

    // print<<<1, 1>>>(d_outdegree, V);
    // print<<<1, 1>>>(d_startidx, V);

    // printEdge<<<1, 1>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V);
    //   Result of  MST

    cudaMalloc(&d_ans, sizeof(long int));
    cudaMemset(d_ans, 0, sizeof(long int));

    // int *flag;
    // cudaHostAlloc(&flag, sizeof(int), 0);

    while (true)
    {

        block = ceil((float)V / 1024);
        // cout << "V: " << V << " Block: " << block << endl;
        //  step1: Adding the min edge to the list.
        //  Creating MST worklist
        Edge *d_mstedge;
        cudaMalloc(&d_mstedge, (V) * sizeof(Edge));

        // Union and find array
        int *d_find;
        cudaMalloc(&d_find, (V) * sizeof(int));
        cudaMemset(d_find, 0, (V) * sizeof(int));
        // init<<<block, 1024>>>(d_find, V);

        find_minedge<<<block, 1024>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V, d_mstedge);

        // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

        // Step2: Remove mirrored edges

        removeEdge<<<block, 1024>>>(V, d_mstedge, d_ans);

        // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

        // Step3: Marking the components
        int *d_changed;
        cudaMalloc(&d_changed, sizeof(int));
        markInitComponent<<<block, 1024>>>(V, d_mstedge, d_find);
        int h_changed = 1;
        while (h_changed)
        {
            cudaMemset(d_changed, 0, sizeof(int));
            markComponent<<<block, 1024>>>(V, d_mstedge, d_find, d_changed);
            // cudaDeviceSynchronize();
            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        }
        // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

        // Step4: Create new vertex ids
        int *d_prefix_sum;
        int *d_block_sum;
        int *d_prefix_sum1;
        cudaMalloc(&d_prefix_sum1, V * sizeof(int));
        cudaMemset(d_prefix_sum1, 0, V * sizeof(int));
        cudaMalloc(&d_block_sum, block * sizeof(int));
        cudaMemset(d_block_sum, 0, block * sizeof(int));
        cudaMalloc(&d_prefix_sum, V * sizeof(int));
        cudaMemset(d_prefix_sum, 0, V * sizeof(int));

        mark_newvertice<<<block, 1024>>>(V, d_find, d_prefix_sum);

        block_prefix_sum_kernel<<<block, 1024, sizeof(int) * 1024>>>(d_prefix_sum, d_prefix_sum, d_block_sum, V);
        scan_block_sums_kernel<<<1, block, sizeof(int) * block>>>(d_block_sum, d_block_sum, block);
        fix_prefix_sum_kernel<<<block, 1024>>>(d_prefix_sum, d_block_sum, V);

        int new_V;
        cudaMemcpy(&new_V, d_prefix_sum + V - 1, sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "New V: " << new_V << endl;

        if (new_V == 1)
        {
            break;
        }
        shift<<<block, 1024>>>(d_prefix_sum, V, d_prefix_sum1);
        d_prefix_sum = d_prefix_sum1;
        // cudaDeviceSynchronize();
        //  cout << "Creating new vertex id step\n";
        //  print_prefsum<<<1, 1>>>(V, d_prefix_sum);

        // Step5: Count the number of edges in the new graph
        int *d_new_startidx;
        int *d_new_outdegree;
        int *d_new_dest_arr;
        int *d_new_weight_arr;

        cudaMalloc(&d_new_startidx, new_V * sizeof(int));
        cudaMalloc(&d_new_outdegree, new_V * sizeof(int));
        cudaMemset(d_new_outdegree, 0, new_V * sizeof(int));
        cudaMalloc(&d_new_dest_arr, 2 * E * sizeof(int));
        cudaMalloc(&d_new_weight_arr, 2 * E * sizeof(int));

        count_outdegree<<<block, 1024>>>(d_startidx, d_outdegree, d_dest_arr, V, d_new_outdegree, d_find, d_prefix_sum);

        find_startIndex<<<block, 1024>>>(d_new_outdegree, new_V, d_new_startidx);
        block_prefix_sum_kernel<<<block, 1024, sizeof(int) * 1024>>>(d_new_startidx, d_new_startidx, d_block_sum, new_V);
        scan_block_sums_kernel<<<1, block, sizeof(int) * block>>>(d_block_sum, d_block_sum, block);
        fix_prefix_sum_kernel<<<block, 1024>>>(d_new_startidx, d_block_sum, new_V);
        int *d_new_startidx1;
        cudaMalloc(&d_new_startidx1, new_V * sizeof(int));
        shift<<<block, 1024>>>(d_new_startidx, new_V, d_new_startidx1);
        // print<<<1, 1>>>(d_new_startidx, new_V);
        addEdges<<<block, 1024>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V, d_prefix_sum, d_new_startidx1, d_new_dest_arr, d_new_weight_arr, d_find);
        cudaMemset(d_new_startidx1, 0, new_V * sizeof(int));
        shift<<<block, 1024>>>(d_new_startidx, new_V, d_new_startidx1);
        d_new_startidx = d_new_startidx1;

        // cudaDeviceSynchronize();
        //  cout << "\nstep 6\n";
        //  print<<<1, 1>>>(d_new_startidx1, new_V);
        //  printEdge<<<1, 1>>>(d_new_startidx, d_new_outdegree, d_new_dest_arr, d_new_weight_arr, new_V);

        V = new_V;
        d_startidx = d_new_startidx;
        d_outdegree = d_new_outdegree;
        d_dest_arr = d_new_dest_arr;
        d_weight_arr = d_new_weight_arr;
        cudaDeviceSynchronize();
        // cout << "\nNext Round\n";
    }
    // Final memcpy

    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(h_ans, d_ans, sizeof(long int), cudaMemcpyDeviceToHost);
    cout << "Final Answer: " << *h_ans << endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // Print only the total MST weight

    // cout << elapsed1.count() << " s\n";

    cudaFree(d_startidx);
    cudaFree(d_outdegree);
    cudaFree(d_dest_arr);
    cudaFree(d_weight_arr);
    cudaFree(d_ans);
    cudaFree(d_graph);
    cudaFree(d_startidx1);

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