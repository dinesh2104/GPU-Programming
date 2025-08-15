import random

def generate_test_case(n, num_edges, num_shelters, num_populated, max_distance_elderly, filename="test_case.txt"):
    edges = set()
    edge_pairs = set()
    all_possible_edges = {(u, v) for u in range(n) for v in range(u + 1, n)}
    
    # Ensure the graph is connected using a spanning tree
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        length = random.randint(1, 1000)
        capacity = random.randint(1, 10000)
        edges.add((u, v, length, capacity))
        edge_pairs.add((u, v))
        all_possible_edges.discard((u, v))
        all_possible_edges.discard((v, u))
    
    print("total Edges:" + str(num_edges))
    print("created spanning tree of edges: " + str(len(edges)))

    # Add remaining edges from the set of possible edges
    while len(edges) < num_edges and all_possible_edges:
        u, v = all_possible_edges.pop()
        length = random.randint(1, 1000)
        capacity = random.randint(1, 1000)
        edges.add((u, v, length, capacity))
        edge_pairs.add((u, v))

    print("created rest of the edges")
    
    # Allocate unique population to cities
    populated_cities = set()
    populated_locations = set()
    while len(populated_locations) < num_populated:
        populated_locations.add(random.randint(0, n-1))
    total_population = 0
    for city in populated_locations:
        prime_age = random.randint(0, 10000)
        elderly = random.randint(0, 10000)
        total_population += prime_age + elderly
        populated_cities.add((city, prime_age, elderly))

    print("Populated Cities")
    
    # Allocate unique shelters' capacity
    shelters = set()
    shelter_cities = set()
    while len(shelter_cities) < num_shelters:
        shelter_cities.add(random.randint(0, n-1))
    shelter_cities = list(shelter_cities)

    allocated_capacity = 0
    threshold = random.randint(50, 200)  # Small threshold to avoid exact allocation
    for i, city in enumerate(shelter_cities[:-1]):
        capacity = random.randint(total_population // num_shelters, total_population // num_shelters + 500)
        
        allocated_capacity += capacity
        shelters.add((city, capacity))

    # Assign remaining capacity to the last shelter
    
    remaining_capacity = (total_population - allocated_capacity) + threshold
    if(remaining_capacity >= 0):
        allocated_capacity += remaining_capacity
        shelters.add((shelter_cities[-1], remaining_capacity))

    print("Total Population: " + str(total_population))
    print("Assigned Shelters Capacity: "+ str(allocated_capacity))
    
    # Save to file
    with open(filename, "w") as file:
        file.write(f"{n}\n")
        file.write(str(len(edges)) + "\n")
        file.write("\n".join(f"{u} {v} {length} {capacity}" for u, v, length, capacity in edges) + "\n")
        file.write(str(len(shelters)) + "\n")
        file.write("\n".join(f"{city} {capacity}" for city, capacity in shelters) + "\n")
        file.write(str(len(populated_cities)) + "\n")
        file.write("\n".join(f"{city} {prime_age} {elderly}" for city, prime_age, elderly in populated_cities) + "\n")
        file.write(f"{max_distance_elderly}\n")

# Example usage
n = 10000
generate_test_case(n, num_edges=random.randint(n - 1, min(n * (n - 1), 1000000)),  num_shelters=1000, num_populated=1000, max_distance_elderly=random.randint(1, 5000))
