# Semester Project - PakTravel AI System
# Part 1: Route Finding with Search Algorithms
# required libraries: pip install networkx matplotlib

import heapq
import time
import networkx as nx
import matplotlib.pyplot as plt

print("=" * 65)
print("     PakTravel AI System - Part 1: Route Finder")
print("=" * 65)

# ============================================================
# TASK 1: Build the Road Network
# ============================================================
print("\n--- TASK 1: Road Network Setup ---\n")

# representing the road network as a weighted adjacency dictionary
# each city key maps to a list of (neighbour_city, distance_km) tuples
# undirected graph so every road appears twice (both directions)
road_network = {
    'Karachi':     [('Hyderabad', 160), ('Quetta', 700)],
    'Hyderabad':   [('Karachi', 160),   ('Sukkur', 380)],
    'Sukkur':      [('Hyderabad', 380), ('Larkana', 80), ('Bahawalpur', 400), ('Multan', 320)],
    'Larkana':     [('Sukkur', 80)],
    'Bahawalpur':  [('Sukkur', 400),    ('Multan', 180)],
    'Multan':      [('Sukkur', 320),    ('Bahawalpur', 180), ('Lahore', 340),
                    ('Faisalabad', 270),('Quetta', 570)],
    'Lahore':      [('Multan', 340),    ('Faisalabad', 130), ('Gujranwala', 80), ('Islamabad', 380)],
    'Faisalabad':  [('Multan', 270),    ('Lahore', 130),     ('Gujranwala', 90)],
    'Gujranwala':  [('Lahore', 80),     ('Faisalabad', 90),  ('Sialkot', 70)],
    'Sialkot':     [('Gujranwala', 70)],
    'Islamabad':   [('Lahore', 380),    ('Rawalpindi', 15),  ('Peshawar', 170)],
    'Rawalpindi':  [('Islamabad', 15),  ('Peshawar', 155)],
    'Peshawar':    [('Islamabad', 170), ('Rawalpindi', 155)],
    'Quetta':      [('Karachi', 700),   ('Multan', 570),     ('Gwadar', 650)],
    'Gwadar':      [('Quetta', 650)],
}

# cities where passengers can switch buses
relay_points = ['Sukkur', 'Multan', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad']

cities = list(road_network.keys())
print(f"Total Cities: {len(cities)}")
print("Cities:", ', '.join(cities))
print(f"\nRelay Points: {', '.join(relay_points)}")

# print all roads (avoid printing duplicates)
print("\nAll Roads in Network:")
seen_roads = set()
road_count = 0
for city, neighbours in road_network.items():
    for neighbour, dist in neighbours:
        key = tuple(sorted([city, neighbour]))
        if key not in seen_roads:
            print(f"  {city:<15} <-> {neighbour:<15} : {dist} km")
            seen_roads.add(key)
            road_count += 1
print(f"\nTotal Roads: {road_count}")


# ============================================================
# TASK 2: Uniform Cost Search (UCS)
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 2: UCS - Cheapest Route ---\n")

def ucs(graph, start, goal):
    """
    Uniform Cost Search - finds the minimum cost (shortest distance) path.
    Uses a priority queue (min-heap) ordered by cumulative cost.
    Returns (path, total_distance, nodes_explored)
    """
    # each entry in the heap: (total_cost, current_city, path_so_far)
    priority_queue = [(0, start, [start])]
    visited = set()
    nodes_explored = 0

    while priority_queue:
        cost, current, path = heapq.heappop(priority_queue)

        # skip if already visited (we already found cheapest way to get here)
        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # found the goal!
        if current == goal:
            return path, cost, nodes_explored

        # expand neighbours
        for neighbour, dist in graph.get(current, []):
            if neighbour not in visited:
                heapq.heappush(priority_queue, (cost + dist, neighbour, path + [neighbour]))

    # if we get here no path was found
    return None, float('inf'), nodes_explored


# test on the 3 required journeys
test_journeys = [
    ('Karachi', 'Multan',    "Ali's Journey"),
    ('Karachi', 'Peshawar',  "Karachi to Peshawar"),
    ('Gwadar',  'Islamabad', "Gwadar to Islamabad"),
]

ucs_results = {}  # save results for task 5 comparison

for start, goal, label in test_journeys:
    path, dist, explored = ucs(road_network, start, goal)
    ucs_results[(start, goal)] = (path, dist, explored)

    print(f"Journey: {label} ({start} -> {goal})")
    if path:
        print(f"  Path Taken  : {' -> '.join(path)}")
        print(f"  Total Dist  : {dist} km")
        stops = len(path) - 2  # exclude start and end
        print(f"  Stops       : {stops} intermediate city/cities")
        print(f"  Nodes Expl. : {explored}")
    else:
        print("  Result: No path found!")
    print()


# ============================================================
# TASK 3: A* Search
# ============================================================
print("=" * 65)
print("--- TASK 3: A* Search ---\n")

# straight-line distances to Islamabad - used as the heuristic h(n)
# heuristic must be admissible (never overestimates actual distance)
heuristic_to_islamabad = {
    'Karachi':    1400, 'Hyderabad': 1250, 'Sukkur':    900,
    'Larkana':     920, 'Bahawalpur': 620, 'Multan':    550,
    'Lahore':      380, 'Faisalabad': 300, 'Gujranwala': 280,
    'Sialkot':     300, 'Islamabad':    0, 'Rawalpindi':  15,
    'Peshawar':    170, 'Quetta':     750, 'Gwadar':    1500,
}

def astar(graph, start, goal, h):
    """
    A* Search - uses f(n) = g(n) + h(n) to guide search toward goal.
    g(n) = actual cost from start, h(n) = heuristic estimate to goal.
    Should explore fewer nodes than UCS because heuristic guides it.
    """
    # heap entry: (f_cost, g_cost, current_city, path)
    pq = [(h.get(start, 0), 0, start, [start])]
    visited = set()
    nodes_explored = 0

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        if current == goal:
            return path, g, nodes_explored

        for neighbour, dist in graph.get(current, []):
            if neighbour not in visited:
                new_g = g + dist
                new_f = new_g + h.get(neighbour, 0)
                heapq.heappush(pq, (new_f, new_g, neighbour, path + [neighbour]))

    return None, float('inf'), nodes_explored


# run and time both algorithms on Karachi -> Islamabad
t_start = time.perf_counter()
as_path, as_dist, as_nodes = astar(road_network, 'Karachi', 'Islamabad', heuristic_to_islamabad)
as_time = (time.perf_counter() - t_start) * 1000

t_start = time.perf_counter()
uc_path, uc_dist, uc_nodes = ucs(road_network, 'Karachi', 'Islamabad')
uc_time = (time.perf_counter() - t_start) * 1000

print("Test Journey: Karachi -> Islamabad\n")
print(f"A* Result:")
print(f"  Path    : {' -> '.join(as_path)}")
print(f"  Distance: {as_dist} km")
print(f"  Nodes   : {as_nodes}  |  Time: {as_time:.4f} ms")

print(f"\nUCS Result:")
print(f"  Path    : {' -> '.join(uc_path)}")
print(f"  Distance: {uc_dist} km")
print(f"  Nodes   : {uc_nodes}  |  Time: {uc_time:.4f} ms")

print(f"\nComparison:")
print(f"  A* explored {uc_nodes - as_nodes} fewer nodes than UCS")
if uc_nodes > as_nodes:
    print("  A* is more efficient! The heuristic guided it away from wrong paths.")
else:
    print("  Both explored similar nodes (small graph, heuristic advantage is limited).")


# ============================================================
# TASK 4: Relay Route Planner
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 4: Relay Route Planner ---\n")

def find_relay_route(start, destination, relay_pts, graph):
    """
    Finds the best 2-leg relay journey: start -> relay_city -> destination.
    Tries all valid relay points and picks the one with minimum total distance.
    Returns: (best_relay_city, leg1_info, leg2_info, total_relay_distance)
    """
    best_relay   = None
    best_total   = float('inf')
    best_leg1    = None
    best_leg2    = None

    for relay in relay_pts:
        # skip if relay is same as start or destination
        if relay == start or relay == destination:
            continue

        # leg 1: start -> relay city
        path1, dist1, _ = ucs(graph, start, relay)
        if path1 is None:
            continue  # no path to this relay point

        # leg 2: relay city -> destination
        path2, dist2, _ = ucs(graph, relay, destination)
        if path2 is None:
            continue  # no path from relay to destination

        total = dist1 + dist2
        if total < best_total:
            best_total = total
            best_relay = relay
            best_leg1  = (path1, dist1)
            best_leg2  = (path2, dist2)

    return best_relay, best_leg1, best_leg2, best_total


# test on the required 4 journeys
relay_tests = [
    ('Karachi', 'Multan',    'via Sukkur'),
    ('Karachi', 'Lahore',    'via Multan'),
    ('Gwadar',  'Islamabad', 'via Quetta'),
    ('Karachi', 'Peshawar',  'via Lahore'),
]

for start, dest, hint in relay_tests:
    relay, leg1, leg2, relay_total = find_relay_route(start, dest, relay_points, road_network)
    direct_path, direct_dist, _ = ucs(road_network, start, dest)

    print(f"Journey: {start} -> {dest}  [{hint}]")
    if relay:
        print(f"  Best Relay City : {relay}")
        print(f"  Leg 1 : {' -> '.join(leg1[0])}  ({leg1[1]} km)")
        print(f"  Leg 2 : {' -> '.join(leg2[0])}  ({leg2[1]} km)")
        print(f"  Relay Total     : {relay_total} km")
        if direct_path:
            diff = relay_total - direct_dist
            print(f"  Direct Route    : {direct_dist} km  ({' -> '.join(direct_path)})")
            print(f"  Extra km (relay vs direct): +{diff} km")
        else:
            print("  Direct Route    : Not Available")
    else:
        print("  No relay route found.")
    print()


# ============================================================
# TASK 5: Full Comparison + Visualization
# ============================================================
print("=" * 65)
print("--- TASK 5: Full Comparison + Visualization ---\n")

def bidirectional_ucs(graph, start, goal):
    """
    Bidirectional UCS - runs two simultaneous searches from start and goal.
    They meet somewhere in the middle. Should explore fewer nodes overall.
    NOTE: this simple version stops at first meeting point so may not
    always give optimal path - the proper version needs more complex bookkeeping
    """
    if start == goal:
        return [start], 0, 1

    # forward search from start
    f_pq = [(0, start, [start])]
    f_visited = {}  # city -> (cost, path)

    # backward search from goal
    b_pq = [(0, goal, [goal])]
    b_visited = {}  # city -> (cost, path)

    nodes_explored = 0

    while f_pq or b_pq:
        # --- expand one node from the forward frontier ---
        if f_pq:
            f_cost, f_node, f_path = heapq.heappop(f_pq)
            if f_node not in f_visited:
                f_visited[f_node] = (f_cost, f_path)
                nodes_explored += 1

                # check if backward search already reached this node
                if f_node in b_visited:
                    b_cost, b_path = b_visited[f_node]
                    # combine: forward path + reversed backward path (excluding meeting node)
                    full_path = f_path + b_path[-2::-1]
                    return full_path, f_cost + b_cost, nodes_explored

                for neighbour, dist in graph.get(f_node, []):
                    if neighbour not in f_visited:
                        heapq.heappush(f_pq, (f_cost + dist, neighbour, f_path + [neighbour]))

        # --- expand one node from the backward frontier ---
        if b_pq:
            b_cost, b_node, b_path = heapq.heappop(b_pq)
            if b_node not in b_visited:
                b_visited[b_node] = (b_cost, b_path)
                nodes_explored += 1

                # check if forward search already reached this node
                if b_node in f_visited:
                    f_cost, f_path = f_visited[b_node]
                    full_path = f_path + b_path[-2::-1]
                    return full_path, b_cost + f_cost, nodes_explored

                for neighbour, dist in graph.get(b_node, []):
                    if neighbour not in b_visited:
                        heapq.heappush(b_pq, (b_cost + dist, neighbour, b_path + [neighbour]))

    return None, float('inf'), nodes_explored


# run all 3 algorithms on Karachi -> Islamabad and collect results
comparison = []

algos = [
    ('UCS',           lambda: ucs(road_network, 'Karachi', 'Islamabad')),
    ('A*',            lambda: astar(road_network, 'Karachi', 'Islamabad', heuristic_to_islamabad)),
    ('Bidirectional', lambda: bidirectional_ucs(road_network, 'Karachi', 'Islamabad')),
]

for name, func in algos:
    t0 = time.perf_counter()
    path, dist, nodes = func()
    elapsed = (time.perf_counter() - t0) * 1000
    comparison.append((name, path, dist, nodes, elapsed))

# print comparison table
print("Algorithm Comparison: Karachi -> Islamabad\n")
print(f"{'Algorithm':<15} {'Distance':>10} {'Nodes':>8} {'Time(ms)':>10}  Path")
print("-" * 95)
for name, path, dist, nodes, ms in comparison:
    path_str = ' -> '.join(path) if path else 'N/A'
    print(f"{name:<15} {dist:>10} {nodes:>8} {ms:>10.4f}  {path_str}")

# --- NetworkX Visualization ---
print("\nBuilding road network visualization with NetworkX...")

G = nx.Graph()
for city, neighbours in road_network.items():
    for nb, dist in neighbours:
        if not G.has_edge(city, nb):
            G.add_edge(city, nb, weight=dist)

# approximate geographic coordinates (lon, lat) for layout
pos = {
    'Karachi':    (67.0, 24.9), 'Hyderabad':  (68.4, 25.4),
    'Sukkur':     (68.9, 27.7), 'Larkana':    (68.2, 27.6),
    'Quetta':     (67.0, 30.2), 'Gwadar':     (62.3, 25.1),
    'Bahawalpur': (71.7, 29.4), 'Multan':     (71.5, 30.2),
    'Faisalabad': (73.1, 31.4), 'Lahore':     (74.3, 31.5),
    'Gujranwala': (74.2, 32.2), 'Sialkot':    (74.5, 32.5),
    'Islamabad':  (73.1, 33.7), 'Rawalpindi': (73.1, 33.55),
    'Peshawar':   (71.5, 34.0),
}

# get A* best path edges
best_path = comparison[1][1]  # A* path
path_edges = []
if best_path:
    for i in range(len(best_path) - 1):
        path_edges.append((best_path[i], best_path[i+1]))

other_edges = [e for e in G.edges() if e not in path_edges and (e[1], e[0]) not in path_edges]

fig, ax = plt.subplots(figsize=(15, 10))

# draw regular edges
nx.draw_networkx_edges(G, pos, edgelist=other_edges,
                       edge_color='#AAAAAA', width=1.5, ax=ax)
# draw highlighted path edges
if path_edges:
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='crimson', width=3.5, ax=ax)

# draw different node types
normal_cities = [c for c in G.nodes() if c not in relay_points and c not in ['Karachi', 'Islamabad']]
nx.draw_networkx_nodes(G, pos, nodelist=normal_cities,
                       node_color='#87CEEB', node_size=450, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[r for r in relay_points if r not in ['Karachi', 'Islamabad']],
                       node_color='#FFA500', node_size=550, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=['Karachi'],
                       node_color='#00CC44', node_size=700, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=['Islamabad'],
                       node_color='#CC0000', node_size=700, ax=ax)

nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

edge_labels = {(u, v): f"{d}km" for u, v, d in G.edges(data='weight')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

ax.set_title(
    "PakTravel Road Network\n"
    "Red path = Karachi→Islamabad (A*)  |  Orange = Relay Points  |  "
    "Green = Karachi  |  Red = Islamabad",
    fontsize=12
)
ax.axis('off')
plt.tight_layout()
plt.savefig('road_network.png', dpi=150, bbox_inches='tight')
print("Map saved as road_network.png")
plt.show()

# written conclusion
print("\n" + "=" * 65)
print("CONCLUSION: Which Algorithm is Best for PakTravel?")
print("=" * 65)
print("""
Scenario    : PakTravel needs to find routes across 15 Pakistani cities.
              The road network is small but will grow over time.

UCS         : Explores all nodes in cost order. Guaranteed optimal path.
              No extra info needed. But explores many unnecessary nodes.

A*          : Uses straight-line distance heuristic to guide the search.
              Explores fewer nodes while still guaranteeing optimal path.
              Best balance of speed and accuracy for PakTravel.

Bidirectional: Starts searching from both ends simultaneously.
              Can be very fast on large graphs but tricky to implement
              correctly for guaranteed optimality.

RECOMMENDATION:
  PakTravel should use A* Search. It is optimal, faster than UCS,
  and scales well as more cities are added. The straight-line distances
  between Pakistani cities are a good admissible heuristic. As long as
  we update the heuristic table when new cities are added, A* will
  remain the most practical choice for the route finder app.
""")