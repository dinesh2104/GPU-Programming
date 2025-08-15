#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>

// TODO: Convert floyd algo to sssp.

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

struct Edge
{
    int src, dest, length, capacity;
    bool is_blocked;
};

struct population
{
    int current_city;
    int prime_age;
    int elderly;
    int max_distance_elderly;
    int time_taken;
    int path[];
};

// Initializes the path and path size
__global__ void
init(long long *wlist, long long *path_size, long long *path, int num_cities, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        path_size[tid] = 1;
        path[tid * (10 * num_cities)] = wlist[tid];
    }
}

__global__ void print(int *dist, int *next, int num_cities)
{
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            printf("%d ", dist[i * num_cities + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            printf("%d ", next[i * num_cities + j]);
        }
        printf("\n");
    }
}

__global__ void printKernel(long long *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%lld ", arr[i]);
    }
    printf("\n");
}

__global__ void initKernel(long long *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }
}

// TODO: missing case when to move the old age
// TODO: Need to calculate the time.
__global__ void compute(int *road, int *wlist_size, long long *wlist, long long *shelter_capacity, long long *pop_primeage,
                        long long *pop_elderly, long long *max_distance_elderly, int *next, int *dist, int num_cities, int *flag, int num_roads, long long *ans_pop_elderly, long long *ans_pop_primeage, long long *time, long long *path_size, long long *paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *wlist_size)
    {
        int city = wlist[tid];

        if (city == -1)
            return;
        // If the city is shelter then move the population to the shelter array ans_pop_elderly and ans_pop_primeage
        if (shelter_capacity[city] > 0)
        {
            // Handling the elders first
            if (shelter_capacity[city] >= pop_elderly[city])
            {
                shelter_capacity[city] -= pop_elderly[city];
                ans_pop_elderly[city] += pop_elderly[city];
                pop_elderly[city] = 0;
            }
            else
            {
                pop_elderly[city] -= shelter_capacity[city];
                ans_pop_elderly[city] += shelter_capacity[city];
                shelter_capacity[city] = 0;
            }
            // Handling the prime age
            if (shelter_capacity[city] >= pop_primeage[city])
            {
                shelter_capacity[city] -= pop_primeage[city];
                ans_pop_primeage[city] += pop_primeage[city];
                pop_primeage[city] = 0;
            }
            else
            {
                pop_primeage[city] -= shelter_capacity[city];
                ans_pop_primeage[city] += shelter_capacity[city];
                shelter_capacity[city] = 0;
            }
        }
        __syncthreads();

        // Check if the city is already empty
        if (pop_primeage[city] == 0 && pop_elderly[city] == 0)
        {
            wlist[tid] = -1;
            return;
        }

        int nearby_shelter = -1;
        long long nearby_shelter_distance = INT_MAX;
        // Find the nearby shelters

        // TODO: only one city got it then mark and make each city to get different shelter
        // TODO: update the decision condition in wise manner.

        for (int i = 0; i < num_cities; i++)
        {
            if (shelter_capacity[i] > 0)
            {
                long long distance = dist[city * num_cities + i];
                if (distance < nearby_shelter_distance)
                {
                    nearby_shelter_distance = distance;
                    nearby_shelter = i;
                }
            }
        }
        printf("For city %d found the nearby shelter as %d\n", city, nearby_shelter);
        // check if no nearby shelter is found
        if (nearby_shelter == -1)
        {
            printf("No nearby shelter found for city %d\n", city);
            return;
        }

        // Finding the next node to move....
        int next_node = next[city * num_cities + nearby_shelter];
        // Check if the road is not blocked
        int next_node_capacity = 0;
        int next_node_length = 0;
        int is_blocked = 1;
        int edge_idx;

        // TODO: Need to add the condition for the roads.
        for (int i = 0; i < num_roads; i++)
        {
            if ((road[5 * i] == city && road[5 * i + 1] == next_node) || (road[5 * i] == next_node && road[5 * i + 1] == city))
            {
                if (atomicCAS(&road[5 * i + 4], 0, 1) == 0)
                {
                    is_blocked = 0;
                    next_node_capacity = road[5 * i + 3];
                    next_node_length = road[5 * i + 2];
                    edge_idx = i;
                    break;
                }
            }
        }

        // TODO: check the below condition
        //  checking if next node dist is within the max_distance_elderly
        int is_within_distance = 0;
        if (next_node_length <= max_distance_elderly[city] && shelter_capacity[nearby_shelter] >= pop_elderly[city])
        {
            is_within_distance = 1;
        }

        // updating the movement
        // TODO: pop_primeage and pop_elderly must be in atomicAdd.
        if (!is_blocked)
        {
            if (is_within_distance)
            {
                int total_pop = pop_primeage[city] + pop_elderly[city];
                // Update the time
                int round = ceil((float)total_pop / (float)next_node_capacity);
                time[city] += round * (next_node_length / 5 * 60);

                // Move the population of elder and primeage to the next city
                atomicAdd((long long unsigned *)&pop_primeage[next_node], pop_primeage[city]);
                atomicAdd((long long unsigned *)&pop_elderly[next_node], pop_elderly[city]);
                pop_primeage[city] = 0;
                pop_elderly[city] = 0;
                max_distance_elderly[city] -= next_node_length;
                max_distance_elderly[next_node] = max_distance_elderly[city];
                max_distance_elderly[city] = 0;

                // updating the time of the next node
                atomicMax((long long unsigned *)&time[next_node], time[city]);
                time[city] = 0;

                // updating the path
                paths[tid * (10 * num_cities) + path_size[tid]] = next_node;
                path_size[tid]++;

                printf("Moving both population from city %d to shelter %d\n", city, next_node);
            }
            else
            {
                int total_pop = pop_primeage[city];
                // Update the time
                int round = ceil((float)total_pop / (float)next_node_capacity);
                time[city] += round * (next_node_length / 5 * 60);
                // Move the population of primeage to the next city
                atomicAdd((long long unsigned *)&pop_primeage[next_node], pop_primeage[city]);
                pop_primeage[city] = 0;

                // updating the time of the next node
                atomicMax((long long unsigned *)&time[next_node], time[city]);
                time[city] = 0;

                paths[tid * (10 * num_cities) + path_size[tid]] = next_node;
                path_size[tid]++;

                printf("Moving prime age population from city %d to shelter %d\n", city, next_node);
            }
            wlist[tid] = next_node;
            *flag = 1;
            // Undo the blocking of road
            road[edge_idx * 5 + 4] = 0;
        }
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
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;

    // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2, capacity2, ...]
    int *roads = new int[num_roads * 4];

    for (int i = 0; i < num_roads; i++)
    {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }

    int num_shelters;
    infile >> num_shelters;

    // Store shelters separately
    long long *shelter_city = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];

    for (int i = 0; i < num_shelters; i++)
    {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;

    // Store populated cities separately
    long long *city = new long long[num_populated_cities];
    long long *pop = new long long[num_populated_cities * 2]; // Flattened [prime-age, elderly] pairs

    for (long long i = 0; i < num_populated_cities; i++)
    {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();
    // Convert roads to a graph representation
    int *dist = new int[num_cities * num_cities];
    int *next = new int[num_cities * num_cities];

    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            if (i == j)
            {
                dist[i * num_cities + j] = 0;
                next[i * num_cities + j] = i;
            }
            else
            {
                dist[i * num_cities + j] = INT_MAX;
                next[i * num_cities + j] = -1;
            }
        }
    }

    // Edge *graph = (Edge *)malloc(num_roads * sizeof(Edge));

    int *h_road = (int *)malloc(num_roads * sizeof(Edge) * 5);

    for (int i = 0; i < num_roads; i++)
    {
        long long u = roads[4 * i];
        long long v = roads[4 * i + 1];
        long long length = roads[4 * i + 2];
        long long capacity = roads[4 * i + 3];

        h_road[5 * i] = u;
        h_road[5 * i + 1] = v;
        h_road[5 * i + 2] = length;
        h_road[5 * i + 3] = capacity;
        h_road[5 * i + 4] = 0; // is_blocked

        dist[u * num_cities + v] = length;
        next[u * num_cities + v] = v;

        dist[v * num_cities + u] = length;
        next[v * num_cities + u] = u;
    }

    int *d_road;
    cudaMalloc(&d_road, num_roads * sizeof(Edge) * 5);
    cudaMemcpy(d_road, h_road, num_roads * sizeof(Edge), cudaMemcpyHostToDevice);

    int *d_num_roads;
    cudaMalloc(&d_num_roads, sizeof(int));
    cudaMemcpy(d_num_roads, &num_roads, sizeof(int), cudaMemcpyHostToDevice);

    // COnvert into CSR representation
    // long long *start_idx = new long long[num_cities];
    // long long *outdegree = new long long[num_cities];
    // long long *dest = new long long[num_roads];
    // long long *length = new long long[num_roads];
    // long long *capacity = new long long[num_roads];

    // int pos = 0;
    // for (long long i = 0; i < num_cities; i++)
    // {
    //     start_idx[i] = pos;
    //     outdegree[i] = graph[i].size();
    //     for (long long j = 0; j < outdegree[i]; j++)
    //     {
    //         dest[start_idx[i] + j] = graph[i][j].first;
    //         length[start_idx[i] + j] = graph[i][j].second.first;
    //         capacity[start_idx[i] + j] = graph[i][j].second.second;
    //         pos++;
    //     }
    // }

    // // Creating Device variables
    // long long *d_start_idx, *d_outdegree, *d_dest, *d_length, *d_capacity;
    // cudaMalloc(&d_start_idx, num_cities * sizeof(long long));
    // cudaMalloc(&d_outdegree, num_cities * sizeof(long long));
    // cudaMalloc(&d_dest, num_roads * sizeof(long long));
    // cudaMalloc(&d_length, num_roads * sizeof(long long));
    // cudaMalloc(&d_capacity, num_roads * sizeof(long long));
    // cudaMemcpy(d_start_idx, start_idx, num_cities * sizeof(long long), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_outdegree, outdegree, num_cities * sizeof(long long), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_dest, dest, num_roads * sizeof(long long), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_length, length, num_roads * sizeof(long long), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_capacity, capacity, num_roads * sizeof(long long), cudaMemcpyHostToDevice);

    // Variable for worklist
    int *d_wlist_size;
    cudaMalloc(&d_wlist_size, sizeof(int));
    cudaMemcpy(d_wlist_size, &num_populated_cities, sizeof(int), cudaMemcpyHostToDevice);
    long long *d_wlist;
    cudaMalloc(&d_wlist, num_populated_cities * sizeof(long long));
    cudaMemcpy(d_wlist, city, num_populated_cities * sizeof(long long), cudaMemcpyHostToDevice);

    // Variable foor the shelter information
    long long *h_shelter_capacity = new long long[num_cities];
    for (int i = 0; i < num_cities; i++)
    {
        h_shelter_capacity[i] = 0;
    }
    for (int i = 0; i < num_shelters; i++)
    {
        h_shelter_capacity[shelter_city[i]] = shelter_capacity[i];
    }
    long long *d_shelter_capacity;
    cudaMalloc(&d_shelter_capacity, num_cities * sizeof(long long));
    cudaMemcpy(d_shelter_capacity, h_shelter_capacity, num_cities * sizeof(long long), cudaMemcpyHostToDevice);

    // Device variable for population value in each city
    long long *pop_primeage = new long long[num_cities];
    long long *pop_elderly = new long long[num_cities];
    for (int i = 0; i < num_cities; i++)
    {
        pop_primeage[i] = 0;
        pop_elderly[i] = 0;
    }
    for (int i = 0; i < num_populated_cities; i++)
    {
        pop_primeage[city[i]] = pop[2 * i];
        pop_elderly[city[i]] = pop[2 * i + 1];
    }

    // cout << "pop_primeage: ";
    // for (int i = 0; i < num_cities; i++)
    // {
    //     cout << pop_primeage[i] << " ";
    // }
    // cout << endl;
    // cout << "pop_elderly: ";
    // for (int i = 0; i < num_cities; i++)
    // {
    //     cout << pop_elderly[i] << " ";
    // }

    long long *d_pop_primeage;
    long long *d_pop_elderly;
    cudaMalloc(&d_pop_primeage, num_cities * sizeof(long long));
    cudaMalloc(&d_pop_elderly, num_cities * sizeof(long long));
    cudaMemcpy(d_pop_primeage, pop_primeage, num_cities * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pop_elderly, pop_elderly, num_cities * sizeof(long long), cudaMemcpyHostToDevice);

    // Device arrays for max_distance_elderly
    long long *h_max_distance_elderly = new long long[num_cities];
    for (int i = 0; i < num_cities; i++)
    {
        h_max_distance_elderly[i] = 0;
    }
    for (int i = 0; i < num_populated_cities; i++)
    {
        if (pop_elderly[city[i]] > 0)
            h_max_distance_elderly[city[i]] = max_distance_elderly;
        else
            h_max_distance_elderly[city[i]] = 0;
    }
    long long *d_max_distance_elderly;
    cudaMalloc(&d_max_distance_elderly, num_cities * sizeof(long long));
    cudaMemcpy(d_max_distance_elderly, h_max_distance_elderly, num_cities * sizeof(long long), cudaMemcpyHostToDevice);

    // calling kernel to compute the shortest distance and path
    int *d_dist;
    cudaMalloc(&d_dist, num_cities * num_cities * sizeof(int));
    cudaMemcpy(d_dist, dist, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);
    int *d_next;
    cudaMalloc(&d_next, num_cities * num_cities * sizeof(int));
    cudaMemcpy(d_next, next, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_cities * num_cities + blockSize - 1) / blockSize;
    for (int k = 0; k < num_cities; k++)
    {
        floydWarshall<<<numBlocks, blockSize>>>(d_dist, num_cities, k, d_next);
        cudaDeviceSynchronize();
    }

    // TODO: Check the dist and next array
    // print<<<1, 1>>>(d_dist, d_next, num_cities);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());

    // Array for shelter data
    long long *d_shelter_elder;
    gpuErrchk(cudaMalloc(&d_shelter_elder, num_cities * sizeof(long long)));
    cout << "0\n";
    initKernel<<<1, 1>>>(d_shelter_elder, num_cities);

    long long *d_shelter_prime;
    cudaMalloc(&d_shelter_prime, num_cities * sizeof(long long));
    initKernel<<<1, 1>>>(d_shelter_prime, num_cities);

    // Array for the timecalculation
    long long *d_time;
    cudaMalloc(&d_time, num_cities * sizeof(long long));
    initKernel<<<1, 1>>>(d_time, num_cities);

    // Creating Kernel to do the computation
    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    // variable to store the path.
    long long *d_path_size;
    cudaMalloc(&d_path_size, num_populated_cities * sizeof(long long));
    long long *d_paths;
    cudaMalloc(&d_paths, 10 * num_populated_cities * num_cities * sizeof(long long *));

    // Printing all the variable
    cout << "Printing all the variable\n";
    cout << "d_wlist:";
    printKernel<<<1, 1>>>(d_wlist, num_populated_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_pop_prime_age:";
    printKernel<<<1, 1>>>(d_pop_primeage, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_pop_elderly:";
    printKernel<<<1, 1>>>(d_pop_elderly, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_max_distance_elderly:";
    printKernel<<<1, 1>>>(d_max_distance_elderly, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_shelter_capacity:";
    printKernel<<<1, 1>>>(d_shelter_capacity, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_shelter_elder:";
    printKernel<<<1, 1>>>(d_shelter_elder, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_shelter_prime:";
    printKernel<<<1, 1>>>(d_shelter_prime, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cout << "d_time:";
    printKernel<<<1, 1>>>(d_time, num_cities);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    while (true)
    {
        cudaMemset(d_flag, 0, sizeof(int));
        compute<<<numBlocks, blockSize>>>(d_road, d_wlist_size, d_wlist, d_shelter_capacity, d_pop_primeage,
                                          d_pop_elderly, d_max_distance_elderly, d_next, d_dist, num_cities, d_flag, num_roads, d_shelter_elder, d_shelter_prime, d_time, d_path_size, d_paths);

        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        int flag = 0;
        cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (flag == 0)
            break;

        cout << "d_wlist:";
        printKernel<<<1, 1>>>(d_wlist, num_populated_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_pop_prime_age:";
        printKernel<<<1, 1>>>(d_pop_primeage, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_pop_elderly:";
        printKernel<<<1, 1>>>(d_pop_elderly, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_max_distance_elderly:";
        printKernel<<<1, 1>>>(d_max_distance_elderly, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_shelter_capacity:";
        printKernel<<<1, 1>>>(d_shelter_capacity, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_shelter_elder:";
        printKernel<<<1, 1>>>(d_shelter_elder, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_shelter_prime:";
        printKernel<<<1, 1>>>(d_shelter_prime, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        cout << "d_time:";
        printKernel<<<1, 1>>>(d_time, num_cities);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        int cont = 0;
        cout << "Do you want to continue? (1/0): ";
        cin >> cont;
        if (cont == 0)
            break;
    }

    // set your answer to these variables
    long long *path_size;
    long long **paths;
    long long *num_drops;
    long long ***drops;

    ofstream outfile(argv[2]); // Read input file from command-line argument
    if (!outfile)
    {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }
    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++)
        {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    return 0;
}
