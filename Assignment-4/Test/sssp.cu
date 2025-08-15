#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <limits>
#include <cassert>
#include <algorithm>
#include <stack>

static const int INF = std::numeric_limits<int>::max();

// Simple GPU error‐check macro
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

__host__ __device__ long long pack(int dist, int parent)
{
    return (static_cast<long long>(dist) << 32) | (unsigned int)parent;
}

__host__ __device__ int unpackDist(long long packed)
{
    return static_cast<int>(packed >> 32);
}

__host__ __device__ int unpackParent(long long packed)
{
    return static_cast<int>(packed & 0xFFFFFFFF);
}

__global__ void relaxEdgesKernel(
    const int *d_u,
    const int *d_v,
    const int *d_w,
    long long *d_distParent,
    bool *d_changed,
    int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E)
        return;

    int src = d_u[idx];
    int dst = d_v[idx];
    int wt = d_w[idx];

    long long srcPacked = d_distParent[src];
    int srcDist = unpackDist(srcPacked);

    if (srcDist == INF)
        return;

    int newDist = srcDist + wt;
    long long oldPacked = d_distParent[dst];

    while (true)
    {
        int oldDist = unpackDist(oldPacked);
        if (newDist >= oldDist)
            break;

        long long newPacked = pack(newDist, src);
        long long prev = atomicCAS((unsigned long long *)&d_distParent[dst], oldPacked, newPacked);
        if (prev == oldPacked)
        {
            *d_changed = true;
            break;
        }
        else
        {
            oldPacked = d_distParent[dst]; // Try again
        }
    }
}

// Function to print the path from source to destination using parent array
int printPath(int dest, const std::vector<int> &parent, int source)
{
    if (parent[dest] == -1 && dest != source)
    {
        std::cout << "No path exists";
        return;
    }

    std::stack<int> path;
    int current = dest;

    // Push all vertices from destination to source
    while (current != -1)
    {
        path.push(current);
        if (current == source)
            break;
        current = parent[current];
    }

    // Print the path
    std::cout << "Path: ";

    int cnt = 0;
    int next_node = -1;
    while (!path.empty())
    {
        cnt++;
        if (cnt == 2)
        {
            next_node = path.top();
        }
        std::cout << path.top();
        path.pop();
        if (!path.empty())
            std::cout << " -> ";
    }
    return next_node;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    // --- 1) Read graph into host edge‐lists
    std::ifstream infile(argv[1]);
    assert(infile && "Cannot open file");

    int V, E;
    infile >> V >> E;

    std::vector<int> h_u(2 * E), h_v(2 * E), h_w(2 * E);
    for (int i = 0; i < E; i++)
    {
        int u, v, w;
        infile >> u >> v >> w;
        h_u[2 * i] = u;
        h_v[2 * i] = v;
        h_w[2 * i] = w;

        h_u[2 * i + 1] = v;
        h_v[2 * i + 1] = u;
        h_w[2 * i + 1] = w;
    }
    E *= 2;

    // --- 2) Allocate and copy edges to device
    int *d_u, *d_v, *d_w;
    gpuErrchk(cudaMalloc(&d_u, E * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_v, E * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_w, E * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_u, h_u.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, h_v.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, h_w.data(), E * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for packed distance + parent and change flag
    long long *d_distParent;
    bool *d_changed;
    gpuErrchk(cudaMalloc(&d_distParent, V * sizeof(long long)));
    gpuErrchk(cudaMalloc(&d_changed, sizeof(bool)));

    std::vector<long long> h_distParent(V);
    std::vector<int> h_dist(V), h_parent(V);

    const int BLOCK = 256;
    const int GRID = (E + BLOCK - 1) / BLOCK;

    // Define how many nearby nodes to print paths for
    const int MAX_NEARBY_NODES = 5;

    // 3) For each source s, run Bellman–Ford on GPU
    for (int s = 0; s < V; s++)
    {
        // Initialize packed distance+parent on host
        for (int i = 0; i < V; i++)
        {
            int dist = (i == s ? 0 : INF);
            h_distParent[i] = pack(dist, -1);
        }
        gpuErrchk(cudaMemcpy(d_distParent, h_distParent.data(), V * sizeof(long long), cudaMemcpyHostToDevice));

        // Run up to V-1 iterations of relaxation
        for (int iter = 0; iter < V - 1; iter++)
        {
            bool h_changed = false;
            gpuErrchk(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));

            relaxEdgesKernel<<<GRID, BLOCK>>>(d_u, d_v, d_w, d_distParent, d_changed, E);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
            if (!h_changed)
                break;
        }

        // Copy packed result back
        gpuErrchk(cudaMemcpy(h_distParent.data(), d_distParent, V * sizeof(long long), cudaMemcpyDeviceToHost));
        for (int i = 0; i < V; i++)
        {
            h_dist[i] = unpackDist(h_distParent[i]);
            h_parent[i] = unpackParent(h_distParent[i]);
        }

        // Only print for first 10 sources
        if (s < 10)
        {
            printf("Source: %d\n", s);

            // Print distances for first 10 nodes
            printf("Distances: ");
            for (int i = 0; i < V && i < 10; i++)
            {
                if (h_dist[i] == INF)
                    printf("%d ", -1);
                else
                    printf("%d ", h_dist[i]);
            }
            printf("\n");

            // Print parents for first 10 nodes
            printf("Parents: ");
            for (int i = 0; i < V && i < 10; i++)
            {
                if (h_dist[i] == INF)
                    printf("%d ", -1);
                else if (i == s)
                    printf("%d ", s);
                else
                    printf("%d ", h_parent[i]);
            }
            printf("\n");

            // Find the nodes closest to source (excluding source itself)
            std::vector<std::pair<int, int>> nodesByDistance;
            for (int i = 0; i < V; i++)
            {
                if (i != s && h_dist[i] != INF)
                {
                    nodesByDistance.push_back({h_dist[i], i});
                }
            }

            // Sort by distance
            std::sort(nodesByDistance.begin(), nodesByDistance.end());

            // Print paths to nearby nodes
            printf("Paths to nearby nodes:\n");
            int count = 0;
            for (const auto &pair : nodesByDistance)
            {
                if (count >= MAX_NEARBY_NODES)
                    break;

                int node = pair.second;
                int distance = pair.first;

                printf("To node %d (distance %d): ", node, distance);
                printPath(node, h_parent, s);
                printf("\n");

                count++;
            }
            printf("\n");
        }
    }

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_distParent);
    cudaFree(d_changed);

    return 0;
}