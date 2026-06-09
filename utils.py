import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def g2dag(G: nx.Graph, d: int, k: int, m: int) -> nx.DiGraph:
    DG = G.to_directed()
    n = len(DG.nodes())
    if n<=2:
        return DG
    assert n > k and n > m
    # Ensure only node with low index points to high index
    for e in list(DG.edges):
        if e[0] >= e[1]:
            DG.remove_edge(*e)
        if e[1] - e[0] > d:
            DG.remove_edge(*e)
    # Remove k lowest index nodes' input edge. Randomly link a node if
    # they have not output edges.
    # And remove m highest index nodes' output edges. Randomly link a node if
    # they have not input edges.
    # ( that make DG to DAG)
    DG.remove_node(0)
    DG.remove_node(n - 1)
    n_list = sorted(list(DG.nodes))
    for i in range(k):
        n_idx = n_list[i]
        for e in list(DG.in_edges(n_idx)):
            DG.remove_edge(*e)
        if len(DG.out_edges(n_idx)) == 0:
            DG.add_edge(n_idx, random.random_choice(n_list[k:]))
    for i in range(n - m, n):
        n_idx = n_list[i]
        for e in list(DG.out_edges(n_idx)):
            DG.remove_edge(*e)
        if len(DG.in_edges(n_idx)) == 0:
            DG.add_edge(random.random_choice(n_list[:n - m], n_idx))
    # If the k<index<n-m, and it only has no input edges or output edges,
    # randomly choose a node in k lowest index nodes to link to or
    # choose a node in m highest index nodes to link to it,
    for i in range(k, m - n):
        n_idx = n_list[i]
        if len(DG.in_edges(n_idx)) == 0:
            DG.add_edge(random.random_choice(n_list[:k], n_idx))
        if len(DG.out_edges(n_idx)) == 0:
            DG.add_edge(n_idx, random.random_choice(n_list[n - m:]))
    DG.add_node(0)
    f = True
    for i in n_list:
        f = True
        for ii in range(i):
            if (ii, i) in DG.edges():
                f = False
        if (f):
            DG.add_edge(0, i)

    DG.add_node(n - 1)
    f = True
    for i in n_list:
        f = True
        for ii in range(i, n):
            if (i, ii) in DG.edges():
                f = False
        if (f):
            DG.add_edge(i, n - 1)
    # then you get a random DAG with k inputs and m outputs
    return DG


# Press the green button in the gutter to run the script.

def generate_random_dag(number_of_tasks):
    G = nx.gnp_random_graph(number_of_tasks, 0.5, directed=True)
    DAG = g2dag(G, d=3, k=0, m=0)
    pos = {}
    start = []

    for i in sorted(list(DAG.nodes)):
        vec = [pos[ii][1] for ii in range(i) if (ii, i) in DAG.edges()]
        if len(vec) > 0:
            y = np.min(vec)
        else:
            y = 1
            start.append(i)
        pos[i] = (-i, y - 1)  # the position of each task in graph

  #  plt.figure()
  #  nx.draw(DAG, pos=pos, with_labels=True)
  #  plt.show()

    return DAG
import random
import math

# Define speed as a sinusoidal function with a minimum speed
def calculate_speed(distance_from_intersection, vmax, vmin, max_distance):
    # Speed peaks between intersections and has a minimum value at intersections
    return (vmax + vmin)/2 + (vmax - vmin)/2 * math.sin(math.pi * distance_from_intersection / max_distance)
# Find the nearest intersection and calculate distance for speed adjustment
def nearest_intersection_distance(x, y, street_positions_x, street_positions_y):
    # Find the closest vertical and horizontal streets to calculate distance to intersection
    nearest_x = min(street_positions_x, key=lambda sx: abs(sx - x))
    nearest_y = min(street_positions_y, key=lambda sy: abs(sy - y))
    distance_to_intersection = math.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)
    return distance_to_intersection
# Update the position of a single user based on their speed
def update_single_user_position(xlim,ylim,user_position, user_direction, street_positions_x, street_positions_y, vmax, vmin, street_width, timestep=1.0):
    x, y = user_position
    max_distance = street_width / 2  # Peak speed midway between intersections
    
    # Calculate distance to the nearest intersection
    distance_from_intersection = nearest_intersection_distance(x, y, street_positions_x, street_positions_y)
    speed= vmax/2
    #speed = calculate_speed(distance_from_intersection, vmax, vmin, max_distance)
    
    # Update position based on direction
    if user_direction == "+horizontal":
        x += speed * timestep  # Move along the x-axis
    elif user_direction == "-horizontal":
        x -= speed * timestep  # Move along the x-axis
    elif user_direction == "+vertical":
        y += speed * timestep  # Move along the y-axis
    else:
        y -= speed * timestep  # Move along the y-axis

    # Wrap-around logic
    x = x % xlim  # Wraps around if x exceeds xlim or goes below 0
    y = y % ylim  # Wraps around if y exceeds ylim or goes below 0
    return (x, y)
# Generate initial street and user data
def generate_streets_with_users(N_s_x,N_s_y, num_users, street_width=24, xmax=100, ymax=100, min_spacing=10, max_spacing=30):
    street_positions_y = []
    street_positions_x = []
    current_y = 0
    current_x = 0

    # Generate y-positions for horizontal streets with spacing
    for _ in range(N_s_x):
        if current_y + street_width <= ymax:
            street_positions_y.append(current_y)
            current_y += street_width + random.uniform(min_spacing, max_spacing)
        else:
            break  # Stop if adding more streets would exceed ymax

    # Generate x-positions for vertical streets with spacing
    for _ in range(N_s_y):
        if current_x + street_width <= xmax:
            street_positions_x.append(current_x)
            current_x += street_width + random.uniform(min_spacing, max_spacing)
        else:
            break  # Stop if adding more streets would exceed xmax

    # Generate user initial positions
    user_positions = []
    user_directions = []  # Stores direction as either 'horizontal' or 'vertical'

    for _ in range(num_users):
        #if random.choice([True, False]):  # Randomly decide between horizontal and vertical street
        if random.choice([True,False]):  # Randomly decide between horizontal and vertical street
            # Horizontal street
            y_user = random.choice(street_positions_y) + street_width/2.0
            x_user = random.uniform(0, xmax)
            user_positions.append((x_user, y_user))
            if random.choice([True, False]):
                user_directions.append("+horizontal")
            else:
                user_directions.append("-horizontal")
        else:
            # Vertical street
            x_user = random.choice(street_positions_x) + street_width/2.0
            y_user = random.uniform(0, ymax)
            user_positions.append((x_user, y_user))
            if random.choice([True, False]):
                user_directions.append("+vertical")
            else:
                user_directions.append("-vertical")


    return street_positions_y, street_positions_x, user_positions, user_directions


def generate_random_pos(xlim,ylim):
    x=random.random()*xlim
    y = random.random() * ylim
    return (x,y)

def plot_users_servers(users,servers,xlim,ylim):
    x={}
    y={}
    for m in range(len(users)):
      x[m]=users[m].pos[0]
      y[m] = users[m].pos[1]
    plt.plot(list(x.values()),list(y.values()),'b*',label='Useres')
    x={}
    y={}
    for s in range(len(servers)):
        x[s] = servers[s].pos[0]
        y[s] = servers[s].pos[1]
    plt.plot(list(x.values()), list(y.values()), 'rs', label='Servers')

    plt.xlim((0,xlim))
    plt.ylim((0,ylim))
    plt.legend(loc='upper right')
    plt.show()
