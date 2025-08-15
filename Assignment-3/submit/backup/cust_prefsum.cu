#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MOD 1000000007

using namespace std;
using std::cin;
using std::cout;
struct Edge
{
    int src, dest, weight;
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

//----------------------------------------------------------------------------

__global__ void find_minedge(int *startidx, int *outdegree, int *dest_arr, int *weight_arr, int V, Edge *mstedge, int start_edge)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int v_id = start_edge * blockDim.x + id; // Start from the edge assigned to this thread
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

__global__ void removeEdge(int V, Edge *mstedge, int start_edge, long int *d_ans)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id = start_edge * blockDim.x + id;
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
        //*d_ans = *d_ans % MOD;
    }
}

__global__ void markComponent(int V, Edge *mstedge, int *find, int start_edge, int *flag)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id = start_edge * blockDim.x + id;
    if (id < V)
    {
        if (mstedge[id].src == -1 && mstedge[id].dest == -1)
        {
            find[id] = id;
        }
        else
        {
            int old_val = find[id];
            int new_val = find[mstedge[id].dest];
            if (old_val != new_val)
            {
                find[id] = new_val;
                *flag = 1;
            }
        }
    }
}

__global__ void mark_newvertice(int V, int *find, int start_edge, int *prefix_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id = start_edge * blockDim.x + id;
    if (id < V)
    {
        prefix_sum[find[id]] = 1;
    }
}

__global__ void prefix_sum(int *pref_sum, int V, int *block_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        int tmp;
        for (int off = 1; off < blockDim.x; off *= 2)
        {
            if (threadIdx.x >= off)
            {
                tmp = pref_sum[(blockIdx.x * blockDim.x) + threadIdx.x - off];
            }
            __syncthreads();
            if (threadIdx.x >= off)
            {
                pref_sum[(blockIdx.x * blockDim.x) + threadIdx.x] += tmp;
            }
            __syncthreads();
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            block_sum[blockIdx.x] = pref_sum[(blockIdx.x * blockDim.x) + threadIdx.x];
        }
    }
}

__global__ void fix_prefsum(int *prefix_sum, int V, int *block_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        for (int i = blockIdx.x; i > 0; i--)
        {
            prefix_sum[id] = prefix_sum[id] + block_sum[i - 1];
        }
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

__global__ void print_prefsum(int V, int *prefix_sum)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d ", prefix_sum[i]);
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

__global__ void print(int *arr, int V)
{
    for (int i = 0; i < V; i++)
    {
        printf("%d ", arr[i]);
    }
}

int main()
{
    // Create a sample graph
    int V;
    cin >> V;
    int E;
    cin >> E;
    vector<vector<pair<int, int>>> graph(V);

    Edge *edges = (Edge *)malloc(E * sizeof(Edge));
    int i = 0;

    while (i < E)
    {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt;
        cin >> s;
        // Adding the edge to the vector.

        if (s == "green")
        {
            wt = wt * 2;
        }
        else if (s == "traffic")
        {
            wt = wt * 5;
        }
        else if (s == "dept")
        {
            wt = wt * 3;
        }
        else
        {
            wt = wt;
        }
        graph[u].push_back({v, wt});
        graph[v].push_back({u, wt});
        i++;
    }

    // Builging CSR Data structure
    int *dest_arr = (int *)malloc(2 * E * sizeof(int));
    int *weight_arr = (int *)malloc(2 * E * sizeof(int));

    int *startidx = (int *)malloc(V * sizeof(int));
    int *outdegree = (int *)malloc(V * sizeof(int));

    int pos = 0;
    for (int i = 0; i < V; i++)
    {
        startidx[i] = pos;
        outdegree[i] = graph[i].size();
        for (int j = 0; j < graph[i].size(); j++)
        {
            dest_arr[pos] = graph[i][j].first;
            weight_arr[pos] = graph[i][j].second;
            pos++;
        }
    }
    cout << "Printing the new graph data structure\n";
    for (int i = 0; i < V; i++)
    {
        for (int j = startidx[i]; j < startidx[i] + graph[i].size(); j++)
        {
            cout << i << " " << dest_arr[j] << " " << weight_arr[j] << "\n";
        }
    }

    // cout << "Created graph\n";

    // Copying the graph data structure to device
    int *d_startidx;
    cudaMalloc(&d_startidx, V * sizeof(int));
    cudaMemcpy(d_startidx, startidx, V * sizeof(int), cudaMemcpyHostToDevice);
    int *d_outdegree;
    cudaMalloc(&d_outdegree, V * sizeof(int));
    cudaMemcpy(d_outdegree, outdegree, V * sizeof(int), cudaMemcpyHostToDevice);
    int *d_dest_arr;
    cudaMalloc(&d_dest_arr, 2 * E * sizeof(int));
    cudaMemcpy(d_dest_arr, dest_arr, 2 * E * sizeof(int), cudaMemcpyHostToDevice);
    int *d_weight_arr;
    cudaMalloc(&d_weight_arr, 2 * E * sizeof(int));
    cudaMemcpy(d_weight_arr, weight_arr, 2 * E * sizeof(int), cudaMemcpyHostToDevice);

    // print<<<1,1>>>(d_outdegree,V);

    // Result of  MST
    long int *d_ans;
    long int *h_ans = (long int *)malloc(sizeof(long int));
    cudaMalloc(&d_ans, sizeof(long int));
    cudaMemset(d_ans, 0, sizeof(long int));

    int *flag;
    cudaHostAlloc(&flag, sizeof(int), 0);

    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();
    // Kernel call(s) here

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // cout << "No of SM's: " << prop.multiProcessorCount << endl;
    int cnt_sm = prop.multiProcessorCount;

    // TODO: Check whether the for loop for the block can be removed.
    while (true)
    {

        int block = ceil((float)V / 1024);
        // cout << "V: " << V << "Block: " << block << endl;
        //  step1: Adding the min edge to the list.
        //  Creating MST worklist
        Edge *d_mstedge;
        cudaMalloc(&d_mstedge, (V) * sizeof(Edge));

        // Union and find array
        int *d_find;
        cudaMalloc(&d_find, (V) * sizeof(int));

        init<<<block, 1024>>>(d_find, V);

        for (int i = 0; i < block; i = i + cnt_sm)
        {
            // cout << "i: " << i << endl;
            find_minedge<<<cnt_sm, 1024>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V, d_mstedge, i);
        }

        // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

        // Step2: Remove mirrored edges
        for (int i = 0; i < block; i = i + cnt_sm)
        {
            // cout << "i: " << i << endl;
            removeEdge<<<cnt_sm, 1024>>>(V, d_mstedge, i, d_ans);
        }

        // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

        // Step3: Marking the components
        while (true)
        {
            *flag = 0;
            for (int i = 0; i < block; i = i + cnt_sm)
            {
                // cout << "i: " << i << endl;
                markComponent<<<cnt_sm, 1024>>>(V, d_mstedge, d_find, i, flag);
            }
            cudaDeviceSynchronize();
            if (*flag == 0)
            {
                break;
            }
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
        for (int i = 0; i < block; i = i + cnt_sm)
        {
            mark_newvertice<<<cnt_sm, 1024>>>(V, d_find, i, d_prefix_sum);
        }

        prefix_sum<<<block, 1024>>>(d_prefix_sum, V, d_block_sum);
        fix_prefsum<<<block, 1024>>>(d_prefix_sum, V, d_block_sum);

        int new_V;
        cudaMemcpy(&new_V, d_prefix_sum + V - 1, sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "New V: " << new_V << endl;

        if (new_V == 1)
        {
            break;
        }
        shift<<<block, 1024>>>(d_prefix_sum, V, d_prefix_sum1);
        d_prefix_sum = d_prefix_sum1;
        cudaDeviceSynchronize();
        // cout << "Creating new vertex id step\n";
        // print_prefsum<<<1, 1>>>(V, d_prefix_sum);

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
        prefix_sum<<<block, 1024>>>(d_new_startidx, new_V, d_block_sum);
        fix_prefsum<<<block, 1024>>>(d_new_startidx, new_V, d_block_sum);
        int *d_new_startidx1;
        cudaMalloc(&d_new_startidx1, new_V * sizeof(int));
        shift<<<block, 1024>>>(d_new_startidx, new_V, d_new_startidx1);

        addEdges<<<block, 1024>>>(d_startidx, d_outdegree, d_dest_arr, d_weight_arr, V, d_prefix_sum, d_new_startidx1, d_new_dest_arr, d_new_weight_arr, d_find);
        cudaMemset(d_new_startidx1, 0, new_V * sizeof(int));
        shift<<<block, 1024>>>(d_new_startidx, new_V, d_new_startidx1);
        d_new_startidx = d_new_startidx1;

        cudaDeviceSynchronize();
        // cout << "\nstep 6\n";
        // print<<<1, 1>>>(d_new_startidx1, new_V);
        // printEdge<<<1, 1>>>(d_new_startidx, d_new_outdegree, d_new_dest_arr, d_new_weight_arr, new_V);

        V = new_V;
        d_startidx = d_new_startidx;
        d_outdegree = d_new_outdegree;
        d_dest_arr = d_new_dest_arr;
        d_weight_arr = d_new_weight_arr;
        cudaDeviceSynchronize();
        // cout << "\nNext Round\n";
    }
    // Final memcpy
    cudaMemcpy(h_ans, d_ans, sizeof(long int), cudaMemcpyDeviceToHost);

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