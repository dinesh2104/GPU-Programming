import os
import random
from typing import List

def prepare(edges: List[List[int]], num_nodes: int, num_edges: int, folder_path: str, filename: str) -> None:
    """
    Save graph details to file in format:
    <num_nodes> <num_edges>
    u v weight
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, "w") as f:
        f.write(f"{num_nodes} {num_edges}\n")
        for edge in edges:
            f.write(" ".join(map(str, edge)) + "\n")

def generate_random_graph(index: int, num_nodes: int, folder_path: str) -> None:
    """
    Generate a connected undirected graph with weighted edges (no labels).
    """
    edges = []
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    edge_set = set()

    # Step 1: Generate a spanning tree to ensure connectivity
    for j in range(1, num_nodes):
        parent = random.randint(0, j - 1)
        u, w = nodes[parent], nodes[j]
        edge_set.add((min(u, w), max(u, w)))
        edges.append([u, w, random.randint(1, 1000)])

    # Step 2: Add extra random edges
    extra_edges = random.randint(num_nodes, num_nodes * 2)
    while len(edge_set) < extra_edges:
        u, w = random.sample(nodes, 2)
        key = (min(u, w), max(u, w))
        if key not in edge_set:
            edge_set.add(key)
            edges.append([u, w, random.randint(1, 1000)])

    filename = f"test{index + 1}.txt"
    prepare(edges, num_nodes, len(edges), folder_path, filename)

# ===== Configuration =====
num_graphs = 1
node_limit = 50
output_dir = r"E:\IITM\GPU\Assignment-4\Test"

# ===== Run Generator =====
for i in range(num_graphs):
    generate_random_graph(i, node_limit, output_dir)
