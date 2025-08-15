#include <iostream>
#include <vector>
#include <string>
#include <limits>
using namespace std;

struct Edge
{
    int src, dest, weight;
    string label;
};

struct Subset
{
    int parent, rank;
};

// Find with path compression
int find(vector<Subset> &subsets, int i)
{
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

// Union by rank
void unionSets(vector<Subset> &subsets, int u, int v)
{
    int root1 = find(subsets, u);
    int root2 = find(subsets, v);

    if (subsets[root1].rank < subsets[root2].rank)
        subsets[root1].parent = root2;
    else if (subsets[root1].rank > subsets[root2].rank)
        subsets[root2].parent = root1;
    else
    {
        subsets[root2].parent = root1;
        subsets[root1].rank++;
    }
}

void boruvkaMST(int V, vector<Edge> &edges)
{
    vector<Subset> subsets(V);
    vector<int> cheapest(V, -1);
    vector<Edge> MST;
    int numTrees = V;
    int MSTweight = 0;
    int iteration = 1;

    for (int v = 0; v < V; ++v)
    {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    while (numTrees > 1)
    {
        fill(cheapest.begin(), cheapest.end(), -1);

        // Find cheapest edge for each component
        for (int i = 0; i < edges.size(); ++i)
        {
            int set1 = find(subsets, edges[i].src);
            int set2 = find(subsets, edges[i].dest);

            if (set1 == set2)
                continue;

            if (cheapest[set1] == -1 || edges[i].weight < edges[cheapest[set1]].weight)
                cheapest[set1] = i;
            if (cheapest[set2] == -1 || edges[i].weight < edges[cheapest[set2]].weight)
                cheapest[set2] = i;
        }

        // Add cheapest edges to MST
        for (int i = 0; i < V; ++i)
        {
            int e = cheapest[i];
            if (e != -1)
            {
                int set1 = find(subsets, edges[e].src);
                int set2 = find(subsets, edges[e].dest);

                if (set1 != set2)
                {
                    MST.push_back(edges[e]);
                    MSTweight += edges[e].weight;
                    unionSets(subsets, set1, set2);
                    --numTrees;
                }
            }
        }

        cout << "Iteration " << iteration++ << ": Number of components = " << numTrees << endl;
    }

    // Print MST result
    cout << "\nEdges in MST:\n";
    for (const auto &edge : MST)
        cout << edge.src << " - " << edge.dest << " (" << edge.weight << ", " << edge.label << ")\n";
    cout << "Total weight: " << MSTweight << endl;
}

int main()
{
    int V, E;
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    vector<Edge> edges(E);
    cout << "Enter each edge in format: src dest weight label\n";
    for (int i = 0; i < E; ++i)
    {
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight >> edges[i].label;
    }

    boruvkaMST(V, edges);
    return 0;
}
