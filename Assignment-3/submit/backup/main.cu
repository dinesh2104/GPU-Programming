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

// __device__ int findAll(int *parent, int i)
// {
//     if (parent[i] == i)
//         return i;
//     return findAll(parent, parent[i]);
// }

__device__ int findAll(int *parent, int i)
{
    if (parent[i] != i)
    {
        parent[i] = findAll(parent, parent[i]); // Path compression
    }
    return parent[i];
}

//__device__ int lock = 0;

// __device__ void unionAll(int *parent, int u, int v)
// {
//     while (true)
//     {
//         if (atomicCAS(&lock, 0, 1) == 0)
//         {
//             int x = findAll(parent, u);
//             int y = findAll(parent, v);
//             parent[x] = y;
//             atomicExch(&lock, 0);
//             break;
//         }
//     }
// }

__device__ void unionAll(int *parent, int u, int v)
{
    int x = findAll(parent, u);
    int y = findAll(parent, v);
    if (x != y)
        parent[x] = y;
}

__global__ void init(int *find, int V)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        find[id] = id;
    }
}

__global__ void dkernel(Edge *edges, int V, int E, Edge *mstedge, int *find, long int *d_ans)
{
    cg::grid_group grid = cg::this_grid();
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        int ii = 0;
        while (true)
        {

            for (int i = 0; i < E; i++)
            {
                int parent = findAll(find, id);
                if ((edges[i].dest == id || edges[i].src == id) && (mstedge[parent].weight == 0 || mstedge[parent].weight > edges[i].weight))
                {

                    int parent_src = findAll(find, edges[i].src);
                    int parent_dest = findAll(find, edges[i].dest);
                    if (parent_src != parent_dest)
                    {
                        // printf("ii: %d parent: %d id: %d src: %d dest: %d weight: %d\n", ii, parent, id, edges[i].src, edges[i].dest, edges[i].weight);

                        mstedge[parent].src = edges[i].src;
                        mstedge[parent].dest = edges[i].dest;
                        mstedge[parent].weight = edges[i].weight;
                    }
                }
            }

            // grid.sync();

            __syncthreads();
            // update the parent array with the d_ans
            // if (id == 0)
            // {
            //     printf("Printing mstedge\n");
            //     for (int i = 0; i < V; i++)
            //     {
            //         printf("%d %d %d\n", mstedge[i].src, mstedge[i].dest, mstedge[i].weight);
            //     }
            // }
            // __syncthreads();

            // Marking the union in sequencial order
            if (threadIdx.x == 0)
            {
                for (int i = 0; i < V; i++)
                {
                    Edge e = mstedge[i];
                    int parent_src = findAll(find, e.src);
                    int parent_dest = findAll(find, e.dest);
                    if (e.src != -1 && e.dest != -1 && parent_src != parent_dest)
                    {
                        unionAll(find, e.src, e.dest);
                        atomicAdd((long long unsigned *)d_ans, e.weight);
                        *d_ans = *d_ans % MOD;
                    }
                }
            }

            // if (id == 0)
            // {
            //     printf("Printing find\n");
            //     for (int i = 0; i < V; i++)
            //     {
            //         printf("%d ", find[i]);
            //     }
            //     printf("\n");
            // }
            __syncthreads();

            mstedge[id].weight = 0;
            mstedge[id].src = -1;
            mstedge[id].dest = -1;
            // printf("From dkernel d_ans: %ld\n", *d_ans);
            int finish = 1;
            for (int i = 0; i < V; i++)
            {
                if (i != id && findAll(find, i) != findAll(find, id))
                {
                    finish = 0;
                    break;
                }
            }
            if (finish == 1)
            {
                break;
            }
            ii++;
        }
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

int main()
{
    // Create a sample graph
    int V;
    cin >> V;
    int E;
    cin >> E;
    vector<Edge *> edges;

    Edge *edges1 = (Edge *)malloc(E * sizeof(Edge));
    int i = 0;

    while (i < E)
    {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt;
        cin >> s;
        // Adding the edge to the vector.
        edges1[i].src = u;
        edges1[i].dest = v;
        if (s == "green")
        {
            edges1[i].weight = wt * 2;
        }
        else if (s == "traffic")
        {
            edges1[i].weight = wt * 5;
        }
        else if (s == "dept")
        {
            edges1[i].weight = wt * 3;
        }
        else
        {
            edges1[i].weight = wt;
        }
        i++;
    }

    // Building a good data structure for the graph
    vector<vector<pair<int, int>>> graph(V);
    for (int i = 0; i < E; i++)
    {
        graph[edges1[i].src].push_back({edges1[i].dest, edges1[i].weight});
        graph[edges1[i].dest].push_back({edges1[i].src, edges1[i].weight});
    }

    int *startidx = (int *)malloc(V * sizeof(int));
    Edge *edges_rep = (Edge *)malloc(V * V * sizeof(Edge));
    int pos = 0;
    for (int i = 0; i < V; i++)
    {
        startidx[i] = pos;
        for (int j = 0; j < graph[i].size(); j++)
        {
            edges_rep[pos].src = i;
            edges_rep[pos].dest = graph[i][j].first;
            edges_rep[pos].weight = graph[i][j].second;
            pos++;
        }
    }
    cout << "Printing the new graph data structure\n";
    for (int i = 0; i < V; i++)
    {
        for (int j = startidx[i]; j < startidx[i] + graph[i].size(); j++)
        {
            cout << edges_rep[j].src << " " << edges_rep[j].dest << " " << edges_rep[j].weight << endl;
        }
    }

    // cout << "Created graph\n";

    // Creating MST worklist
    Edge *d_mstedge;
    cudaMalloc(&d_mstedge, (V) * sizeof(Edge));

    // Union and find array
    int *d_find;
    cudaMalloc(&d_find, (V) * sizeof(int));

    // Result of  MST
    long int *d_ans;
    long int *h_ans = (long int *)malloc(sizeof(long int));
    cudaMalloc(&d_ans, sizeof(long int));
    cudaMemset(d_ans, 0, sizeof(long int));

    Edge *d_edges;
    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMemcpy(d_edges, edges1, E * sizeof(Edge), cudaMemcpyHostToDevice);

    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();
    // Kernel call(s) here
    int block = ceil((float)V / 1024);
    init<<<block, 1024>>>(d_find, V);

    dkernel<<<block, 1024>>>(d_edges, V, E, d_mstedge, d_find, d_ans);

    // printMst<<<1, 1>>>(d_mstedge, V, d_find, d_ans);

    // print_kernel<<<block, 1024>>>(d_edges, V, E);

    cudaMemcpy(h_ans, d_ans, sizeof(long int), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // Print only the total MST weight

    // cout << elapsed1.count() << " s\n";

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