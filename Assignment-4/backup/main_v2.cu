#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>

// TODO: Convert floyd algo to sssp.

// TODO: Check for variable that can be changed from long long to int....

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

struct population
{
    int current_city;
    int prime_age;
    int elderly;
    int max_distance_elderly;
    int time_taken;
};

// Initializes the path and path size
__global__ void init(long long *wlist, long long *path_size, long long *path, int num_cities, int n)
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

__global__ void printRoad(int *road, int num_roads)
{
    for (int i = 0; i < num_roads; i++)
    {
        printf("Road %d: %d %d %d %d %d\n", i, road[5 * i], road[5 * i + 1], road[5 * i + 2], road[5 * i + 3], road[5 * i + 4]);
    }
}

__global__ void printpath(long long *path, long long *path_size, int num_cities, int num_populated_cities)
{
    for (int i = 0; i < num_populated_cities; i++)
    {
        printf("Path for city %d: ", i);
        for (int j = 0; j < path_size[i]; j++)
        {
            printf("%lld ", path[i * (10 * num_cities) + j]);
        }
        printf("\n");
    }
}

__global__ void printpopulation(population *pop, int num_populated_cities)
{
    for (int i = 0; i < num_populated_cities; i++)
    {
        printf("Population %d: current_city=%d, prime_age=%d, elderly=%d, max_distance_elderly=%d, time_taken=%d\n",
               i, pop[i].current_city, pop[i].prime_age, pop[i].elderly, pop[i].max_distance_elderly, pop[i].time_taken);
    }
}

__global__ void printDrop(long long *drop_count, long long *drop_value, int num_populated_cities)
{
    for (int i = 0; i < num_populated_cities; i++)
    {
        printf("Drop %d: ", i);
        for (int k = 0; k < drop_count[i]; k++)
        {
            printf("[");
            for (int j = 0; j < 3; j++)
            {
                printf("%lld ", drop_value[i * (30 * 3) + k * 3 + j]);
            }
            printf("] ");
        }

        printf("\n");
    }
}

// -------------------------------------------------------------

__global__ void initKernel(long long *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = 0;
    }
}

// Initializing the init path as city path.
__global__ void initPath(long long *path, int num_populated_cities, long long *wlist, int num_cities, long long *path_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_populated_cities)
    {
        path[tid * 10 * num_cities] = wlist[tid];
        // printf("Initial Path for city %lld: %lld\n", wlist[tid], path[tid * 10 * num_cities]);
        path_size[tid] = 1;
    }
}

// TODO: missing case when to move the old age
// TODO: Need to calculate the time.

__device__ volatile int lock[100000] = {0};

__global__ void compute(int *road, int *wlist_size, long long *wlist, long long *shelter_capacity, population *pop_t, int *next, int *dist, int num_cities, int *flag, int num_roads, long long *drop_count, long long *drop_value, long long *path_size, long long *paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *wlist_size)
    {
        int city = wlist[tid];

        if (city == -1)
            return;
        // If the city is shelter then move the population to the shelter array ans_pop_elderly and ans_pop_primeage
        // TODO: Need to implement the sysnc condition for the shelter. Try with single

        if (shelter_capacity[city] > 0)
        {
            // TODO: Spin lock need to moved outside the loop.....
            int oldval = 1;
            while (oldval != 0)
            {
                oldval = atomicCAS((int *)&lock[city], 0, 1);
                if (oldval == 0)
                {

                    int ans_pop_elderly = 0;
                    int ans_pop_primeage = 0;
                    if (shelter_capacity[city] >= pop_t[tid].elderly)
                    {
                        shelter_capacity[city] -= pop_t[tid].elderly;
                        ans_pop_elderly = pop_t[tid].elderly;
                        pop_t[tid].elderly = 0;
                    }
                    else
                    {
                        pop_t[tid].elderly -= shelter_capacity[city];
                        ans_pop_elderly = shelter_capacity[city];
                        shelter_capacity[city] = 0;
                    }

                    // Handling the prime age
                    if (shelter_capacity[city] >= pop_t[tid].prime_age)
                    {
                        shelter_capacity[city] -= pop_t[tid].prime_age;
                        ans_pop_primeage = pop_t[tid].prime_age;
                        pop_t[tid].prime_age = 0;
                    }
                    else
                    {
                        pop_t[tid].prime_age -= shelter_capacity[city];
                        ans_pop_primeage = shelter_capacity[city];
                        shelter_capacity[city] = 0;
                    }
                    if (ans_pop_elderly != 0 || ans_pop_primeage != 0)
                    {
                        printf("Dropping %d population in city %d\n", ans_pop_elderly + ans_pop_primeage, city);
                        drop_value[tid * (30 * 3) + drop_count[tid] * 3] = city;
                        drop_value[tid * (30 * 3) + drop_count[tid] * 3 + 1] = ans_pop_elderly;
                        drop_value[tid * (30 * 3) + drop_count[tid] * 3 + 2] = ans_pop_primeage;
                        drop_count[tid]++;
                    }
                    lock[city] = 0;
                }
            }
        }

        // Check if the city is already empty
        if (pop_t[tid].elderly == 0 && pop_t[tid].prime_age == 0)
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
        //  check if no nearby shelter is found
        if (nearby_shelter == -1)
        {
            // printf("No nearby shelter found for city %d\n", city);
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
            // Check if the road is blocked
            // printf("Checking the road %d %d %d\n", road[5 * i], road[5 * i + 1], road[5 * i + 4]);
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
        if (next_node_length <= pop_t[tid].max_distance_elderly && shelter_capacity[nearby_shelter] >= pop_t[tid].elderly)
        {
            is_within_distance = 1;
        }

        // updating the movement
        // printf("Is the road blocked %d\n", is_blocked);
        // TODO: pop_primeage and pop_elderly must be in atomicAdd.
        if (!is_blocked)
        {
            if (is_within_distance)
            {
                int total_pop = pop_t[tid].prime_age + pop_t[tid].elderly;
                // Update the time
                int round = ceil((float)total_pop / (float)next_node_capacity);
                pop_t[tid].time_taken += round * (next_node_length / 5 * 60);

                // Move the population of elder and primeage to the next city
                pop_t[tid].current_city = next_node;
                pop_t[tid].max_distance_elderly -= next_node_length;

                // updating the path
                paths[tid * (10 * num_cities) + path_size[tid]] = next_node;
                path_size[tid]++;

                printf("Moving both population from city %d to shelter %d\n", city, next_node);
            }
            else
            {
                int total_pop = pop_t[tid].prime_age;
                // Update the time
                int round = ceil((float)total_pop / (float)next_node_capacity);
                pop_t[tid].time_taken += round * (next_node_length / 5 * 60);
                // Move the population of primeage to the next city
                int ans_pop_primeage = 0;
                int ans_pop_elderly = pop_t[tid].elderly;
                if (ans_pop_elderly != 0)
                {
                    drop_value[tid * (30 * 3) + drop_count[tid] * 3] = city;
                    drop_value[tid * (30 * 3) + drop_count[tid] * 3 + 1] = ans_pop_elderly;
                    drop_value[tid * (30 * 3) + drop_count[tid] * 3 + 2] = ans_pop_primeage;
                    drop_count[tid]++;
                }

                pop_t[tid].elderly = 0;

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

    // Code for the road details.......

    int *h_road = (int *)malloc(num_roads * sizeof(int) * 5);

    for (int i = 0; i < num_roads; i++)
    {
        int u = roads[4 * i];
        int v = roads[4 * i + 1];
        int length = roads[4 * i + 2];
        int capacity = roads[4 * i + 3];

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
    cudaMalloc(&d_road, num_roads * sizeof(int) * 5);
    cudaMemcpy(d_road, h_road, num_roads * sizeof(int) * 5, cudaMemcpyHostToDevice);

    int *d_num_roads;
    cudaMalloc(&d_num_roads, sizeof(int));
    cudaMemcpy(d_num_roads, &num_roads, sizeof(int), cudaMemcpyHostToDevice);

    // printRoad<<<1, 1>>>(d_road, num_roads);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();

    // Variable for worklist
    int *d_wlist_size;
    cudaMalloc(&d_wlist_size, sizeof(int));
    cudaMemcpy(d_wlist_size, &num_populated_cities, sizeof(int), cudaMemcpyHostToDevice);
    long long *d_wlist;
    cudaMalloc(&d_wlist, num_populated_cities * sizeof(long long));
    cudaMemcpy(d_wlist, city, num_populated_cities * sizeof(long long), cudaMemcpyHostToDevice);

    // Variable for the shelter information
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

    // Device and host variable for population struct for the populated city.
    population *h_pop;
    h_pop = (population *)malloc(num_populated_cities * sizeof(population));
    for (int i = 0; i < num_populated_cities; i++)
    {
        h_pop[i].current_city = city[i];
        h_pop[i].prime_age = pop[2 * i];
        h_pop[i].elderly = pop[2 * i + 1];
        h_pop[i].max_distance_elderly = max_distance_elderly;
        h_pop[i].time_taken = 0;
    }

    population *d_pop;
    cudaMalloc(&d_pop, num_populated_cities * sizeof(population));
    cudaMemcpy(d_pop, h_pop, num_populated_cities * sizeof(population), cudaMemcpyHostToDevice);

    // TODO: Check whether it is working or not.......

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
    long long *d_drop;
    long long *d_drop_value;
    cudaMalloc(&d_drop, num_populated_cities * sizeof(long long));
    cudaMemset(d_drop, 0, num_populated_cities * sizeof(long long));
    cudaMalloc(&d_drop_value, num_populated_cities * (30 * 3) * sizeof(long long));
    cudaMemset(d_drop_value, 0, num_populated_cities * (30 * 3) * sizeof(long long));

    // Creating Kernel to do the computation
    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    // variable to store the path.
    long long *d_path_size;
    cudaMalloc(&d_path_size, num_populated_cities * sizeof(long long));
    long long *d_paths;
    cudaMalloc(&d_paths, 10 * num_populated_cities * num_cities * sizeof(long long *));
    cudaMemset(d_paths, 0, 10 * num_populated_cities * num_cities * sizeof(long long *));

    int blockSize1 = 256;
    int numBlocks1 = (num_populated_cities + blockSize1 - 1) / blockSize1;

    initPath<<<numBlocks1, blockSize1>>>(d_paths, num_populated_cities, d_wlist, num_cities, d_path_size);
    // Printing all the variable
    // cout << "Printing all the variable\n";
    // cout << "d_wlist:";
    // printKernel<<<1, 1>>>(d_wlist, num_populated_cities);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    // cout << "d_shelter_capacity:";
    // printKernel<<<1, 1>>>(d_shelter_capacity, num_cities);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    // cout << "Drop details\n";
    // printDrop<<<1, 1>>>(d_drop, d_drop_value, num_populated_cities);
    // cout << "Population Details\n";
    // printpopulation<<<1, 1>>>(d_pop, num_populated_cities);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    // cout << "d_road:";
    // printpath<<<1, 1>>>(d_paths, d_path_size, num_cities, num_populated_cities);
    // gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    // printf("=========================================================\n");

    while (true)
    {
        cudaMemset(d_flag, 0, sizeof(int));
        compute<<<numBlocks, blockSize>>>(d_road, d_wlist_size, d_wlist, d_shelter_capacity, d_pop, d_next, d_dist, num_cities, d_flag, num_roads, d_drop, d_drop_value, d_path_size, d_paths);

        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        int flag = 0;
        cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (flag == 0)
            break;

        // cout << "d_wlist:";
        // printKernel<<<1, 1>>>(d_wlist, num_populated_cities);
        // gpuErrchk(cudaPeekAtLastError());
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaGetLastError());
        // cout << "d_shelter_capacity:";
        // printKernel<<<1, 1>>>(d_shelter_capacity, num_cities);
        // gpuErrchk(cudaPeekAtLastError());
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaGetLastError());
        // cout << "Drop details\n";
        // printDrop<<<1, 1>>>(d_drop, d_drop_value, num_populated_cities);
        // cout << "Population Details\n";
        // printpopulation<<<1, 1>>>(d_pop, num_populated_cities);
        // gpuErrchk(cudaPeekAtLastError());
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaGetLastError());
        // cout << "d_road:";
        // printpath<<<1, 1>>>(d_paths, d_path_size, num_cities, num_populated_cities);
        // gpuErrchk(cudaPeekAtLastError());
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaGetLastError());
        printf("=========================================================\n");

        // int cont = 0;
        // cout << "Do you want to continue? (1/0): ";
        // cin >> cont;
        // if (cont == 0)
        //     break;
    }

    // set your answer to these variables
    long long *path_size;
    long long **paths;
    long long *num_drops;
    long long ***drops;

    path_size = new long long[num_populated_cities];
    cudaMemcpy(path_size, d_path_size, num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    paths = new long long *[num_populated_cities];
    for (long long i = 0; i < num_populated_cities; i++)
    {
        paths[i] = new long long[10 * num_cities];
        cudaMemcpy(paths[i], d_paths + i * 10 * num_cities, 10 * num_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    }
    // TODO: Update the below code to get the drops.
    num_drops = new long long[num_populated_cities];
    cudaMemcpy(num_drops, d_drop, num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    drops = new long long **[num_populated_cities];
    for (long long i = 0; i < num_populated_cities; i++)
    {
        drops[i] = new long long *[num_drops[i]];
        for (long long j = 0; j < num_drops[i]; j++)
        {
            drops[i][j] = new long long[3];
            cudaMemcpy(drops[i][j], d_drop_value + i * (30 * 3) + j * 3, 3 * sizeof(long long), cudaMemcpyDeviceToHost);
        }
    }

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
