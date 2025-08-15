#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

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

__global__ void floydWarshall(int *dist, int V, int k, int *next)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = tid / V;
    int jj = tid % V;

    if (ii == k || jj == k)
        return;

    if (ii < V && jj < V)
    {
        if (dist[ii * V + k] != INT_MAX && dist[k * V + jj] != INT_MAX)
        {
            if (dist[ii * V + jj] > dist[ii * V + k] + dist[k * V + jj])
            {
                dist[ii * V + jj] = dist[ii * V + k] + dist[k * V + jj];
                next[ii * V + jj] = next[ii * V + k];
            }

            // printf("dist[%d][%d] = %d\n", ii, jj, dist[ii * V + jj]);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    int V;
    infile >> V;
    int E;
    infile >> E;

    int *distmatrix = (int *)malloc(V * V * sizeof(int));
    int *next = (int *)malloc(V * V * sizeof(int));
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (i == j)
            {
                distmatrix[i * V + j] = 0;
                next[i * V + j] = i;
            }
            else
            {
                distmatrix[i * V + j] = INT_MAX;
                next[i * V + j] = -1;
            }
        }
    }
    int i = 0;

    while (i < E)
    {
        int u1, v1, wt1;

        infile >> u1 >> v1 >> wt1;
        distmatrix[u1 * V + v1] = wt1;
        distmatrix[v1 * V + u1] = wt1;
        next[u1 * V + v1] = v1;
        next[v1 * V + u1] = u1;
        i++;
    }

    printf("Graph:\n");
    // for (int i = 0; i < V; i++)
    // {
    //     for (auto j : graph[i])
    //     {
    //         printf("(%d,%d, %d) ", i, j.first, j.second);
    //     }
    //     printf("\n");
    // }

    printf("Distance Matrix:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (distmatrix[i * V + j] == INT_MAX)
                cout << "INF ";
            else
                cout << distmatrix[i * V + j] << " ";
        }
        cout << endl;
        break;
    }

    int *d_distmatrix, *d_ans;
    cudaMalloc(&d_distmatrix, V * V * sizeof(int));

    int *d_next_node, *d_ans_node;
    d_ans_node = (int *)malloc(V * V * sizeof(int));
    cudaMalloc(&d_next_node, V * V * sizeof(int));
    cudaMemcpy(d_next_node, next, V * V * sizeof(int), cudaMemcpyHostToDevice);

    d_ans = (int *)malloc(V * V * sizeof(int));
    cudaMemcpy(d_distmatrix, distmatrix, V * V * sizeof(int), cudaMemcpyHostToDevice);

    for (int k = 0; k < V; k++)
    {
        int blockSize = 256;
        int numBlocks = (V * V + blockSize - 1) / blockSize;
        floydWarshall<<<numBlocks, blockSize>>>(d_distmatrix, V, k, d_next_node);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaMemcpy(d_ans, d_distmatrix, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Shortest Path Matrix:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V && j < 10; j++)
        {
            if (d_ans[i * V + j] == INT_MAX)
                cout << "INF ";
            else
                cout << d_ans[i * V + j] << " ";
        }
        cout << endl;
        if (i == 10)
            break;
    }
    cudaMemcpy(d_ans_node, d_next_node, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Next Node Matrix:\n");

    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V && j < 10; j++)
        {
            if (d_ans_node[i * V + j] == -1)
                cout << "INF ";
            else
                cout << d_ans_node[i * V + j] << " ";
        }
        cout << endl;
        if (i == 10)
            break;
    }

    // Sequential version for comparison
    // for (int k = 0; k < V; k++)
    // {
    //     for (int i = 0; i < V; i++)
    //     {
    //         for (int j = 0; j < V; j++)
    //         {
    //             if (distmatrix[i * V + k] != INT_MAX && distmatrix[k * V + j] != INT_MAX && i != k && j != k && distmatrix[i * V + k] + distmatrix[k * V + j] < distmatrix[i * V + j])
    //             {
    //                 distmatrix[i * V + j] = distmatrix[i * V + k] + distmatrix[k * V + j];
    //                 next[i * V + j] = next[i * V + k];
    //             }
    //         }
    //     }
    // }

    // printf("Sequential Shortest Path Matrix:\n");
    // for (int i = 0; i < V; i++)
    // {
    //     for (int j = 0; j < V; j++)
    //     {
    //         if (distmatrix[i * V + j] == INT_MAX)
    //             cout << "INF ";
    //         else
    //             cout << distmatrix[i * V + j] << " ";
    //     }
    //     cout << endl;
    //     break;
    // }

    // printf("Sequential Next Node Matrix:\n");
    // for (int i = 0; i < V; i++)
    // {
    //     for (int j = 0; j < V; j++)
    //     {
    //         if (next[i * V + j] == -1)
    //             cout << "INF ";
    //         else
    //             cout << next[i * V + j] << " ";
    //     }
    //     cout << endl;
    // }

    fflush(stdout);

    cudaFree(d_distmatrix);
    cudaFree(d_ans);
    free(distmatrix);
}
