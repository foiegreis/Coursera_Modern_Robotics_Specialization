
import numpy as np
from scipy.special import gammainc
import math


# Uniform sampling in a hyperspere
# Based on Matlab implementation by Roger Stafford
# Can be optimized for Bridson algorithm by excluding all points within the r/2 sphere
def hypersphere_volume_sample(center, radius, k=1):
    ndim = center.size
    x = np.random.normal(size=(k, ndim))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(k, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p


# Uniform sampling on the sphere's surface
def hypersphere_surface_sample(center, radius, k=1):
    ndim = center.size
    vec = np.random.standard_normal(size=(k, ndim))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    p = center + np.multiply(vec, radius)
    return p


def squared_distance(p0, p1):
    return np.sum(np.square(p0 - p1))


def Bridson_sampling(dims=np.array([1.0, 1.0]), radius=0.05, k=30, hypersphere_sample=hypersphere_volume_sample):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    ndim = dims.size

    # size of the sphere from which the samples are drawn relative to the size of a disc (radius)
    sample_factor = 2
    if hypersphere_sample == hypersphere_volume_sample:
        sample_factor = 2

    # for the surface sampler, all new points are almost exactly 1 radius away from at least one existing sample
    # eps to avoid rejection
    if hypersphere_sample == hypersphere_surface_sample:
        eps = 0.001
        sample_factor = 1 + eps

    def in_limits(p):
        return np.all(np.zeros(ndim) <= p) and np.all(p < dims)

    # Check if there are samples closer than "squared_radius" to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)

        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < squared_radius):
            return True

    def add_point(p):
        points.append(p)
        indices = (p / cellsize).astype(int)
        P[tuple(indices)] = p

    cellsize = radius / np.sqrt(ndim)
    gridsize = (np.ceil(dims / cellsize)).astype(int)

    # Squared radius because we'll compare squared distance
    squared_radius = radius * radius

    # Positions of cells
    P = np.empty(np.append(gridsize, ndim), dtype=np.float32)  # n-dim value for each grid cell
    # Initialise empty cells with NaNs
    P.fill(np.nan)

    points = []
    add_point(np.random.uniform(np.zeros(ndim), dims))
    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = hypersphere_sample(np.array(p), radius * sample_factor, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[~np.isnan(P).any(axis=ndim)]




def remove_comments(oldfile, newfile):
    # reading the file
    with open(oldfile) as fp:
        contents = fp.readlines()
        # whenever an element is deleted from the list, the length of the list will be decreased
        decreasing_counter = 0
        for number in range(len(contents)):
            # delete the line if it starts with "#"
            if contents[number-decreasing_counter].startswith("#"):
                contents.remove(contents[number-decreasing_counter])
                decreasing_counter += 1
    # writing into a new file
    with open(newfile, "w") as fp:
        fp.writelines(contents)
    return


def interseg(x1, y1, x2, y2, xc, yc, r):

    E = np.array([[x1], [y1]])
    L = np.array([[x2], [y2]])
    C = np.array([[xc], [yc]])
    d = L - E
    f = E - C

    a = d.T.dot(d)
    b = 2.0 * f.T.dot(d)
    c = f.T.dot(f) - r * r
    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:       # no intersection
        collision = 0
        return collision

    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2.0 * a);
    t2 = (-b + discriminant) / (2.0 * a);
    if t1 >= 0 and t1 <= 1:
        collision = 1               # edge entering the circle
        return collision
    if t2 >= 0 and t2 <= 1:
        collision = 1               # edge exiting the circle
        return collision

    collision = 0               # edge before or passed the circle
    return collision


# sampling uniformly over the entire space
P = Bridson_sampling(dims=np.array([1.0,1.0]), radius=0.125, k=30, hypersphere_sample=hypersphere_volume_sample) - [0.5, 0.5]

remove_comments("obstacles.csv", "obstacles2.csv")
obstacles_file = open("obstacles2.csv")
obstacles_data = np.loadtxt(obstacles_file, delimiter=",")

incircle = np.zeros(len(P))             # identify and delete sampling points inside the circles
for i in range(0, len(P)):
    for j in range(0, len(obstacles_data)):
        circle_x = obstacles_data[j][0]
        circle_y = obstacles_data[j][1]
        rad = obstacles_data[j][2] / 2
        if (P[i][0] - circle_x) * (P[i][0] - circle_x) + (P[i][1] - circle_y) * (P[i][1] - circle_y) <= rad * rad:
            incircle[i] = 1
            continue

for i in range(len(P)):
    if incircle[i] == 1:
        P[i][0] = 100
P = np.delete(P, np.where(P > 99)[0], axis=0)

# create the list of nodes
nodes_array = np.empty((0,4), float)
nodes_array = np.append(nodes_array, np.array([[0, -0.5, -0.5, math.sqrt(2)]]), axis=0)
for i in range(len(P)):
    x = P[i][0]
    y = P[i][1]
    opt = math.sqrt((0.5-x)*(0.5-x)+(0.5-y)*(0.5-y))
    nodes_array = np.append(nodes_array, np.array([[i + 1, x, y, opt]]), axis=0)
nodes_array = np.append(nodes_array, np.array([[len(P)+1, 0.5, 0.5, 0]]), axis=0)

export_nodes = np.copy(nodes_array)     # create nodes.csv file
for i in range(len(export_nodes)):
    export_nodes[i][0] += 1
np.savetxt('nodes.csv', export_nodes, delimiter=',', fmt='%f')

# create the edges with k=3 (3 neighbors per node)
N = len(P) + 2
dist = np.empty((N, N), float)
for i in range(N):
    for j in range(N):
        dx = nodes_array[j][1] - nodes_array[i][1]
        dy = nodes_array[j][2] - nodes_array[i][2]
        dist[i][j] = math.sqrt(dx * dx + dy * dy)

for i in range(N):
    dist[i][i] = 1000
mins = np.argmin(dist, axis=1)
ks = np.empty((N,3))

for i in range(N):
    ks[i][0] = mins[i]
    dist[i][mins[i]] = 1000
mins = np.argmin(dist, axis=1)

for i in range(N):
    ks[i][1] = mins[i]
    dist[i][mins[i]] = 1000
mins = np.argmin(dist, axis=1)

for i in range(N):
    ks[i][2] = mins[i]

for i in range(N):
    for j in range(N):
        dx = nodes_array[j][1] - nodes_array[i][1]
        dy = nodes_array[j][2] - nodes_array[i][2]
        dist[i][j] = math.sqrt(dx * dx + dy * dy)

# create the list of edges
edges = np.empty((0, 3), float)
score = np.zeros((N, N), dtype=float)

for i in range(N):
    for j in range(3):
        ksij = int(ks[i][j])
        if score[ksij][i] > 0 or score[i][ksij] > 0:
            continue
        edges = np.append(edges, np.array([[ksij, i, dist[ksij, i]]]), axis=0)
        score[ksij][i] = 1.0
        score[i][ksij] = 1.0

# identify and delete the edges in collision with the circles
collision_table = np.zeros((len(edges), len(obstacles_data)))
for i in range(len(edges)):
    istart = int(edges[i][0])
    iend = int(edges[i][1])
    for j in range(len(obstacles_data)):
        rc = obstacles_data[j][2] / 2.0
        collision_table[i][j] = interseg(nodes_array[istart][1], nodes_array[istart][2],  \
                                         nodes_array[iend][1], nodes_array[iend][2],    \
                                         obstacles_data[j][0], obstacles_data[j][1], rc)

for i in range(len(edges)-1, 0, -1):
    tags =0
    for j in range(len(obstacles_data)):
        tags += collision_table[i][j]
    if tags > 0:
        edges = np.delete(edges, (i), axis=0)

export_edges = np.copy(edges)           # create the edges.csv file
for i in range(len(export_edges)):
    export_edges[i][0] += 1
    export_edges[i][1] += 1

np.savetxt('edges.csv', export_edges, delimiter=',', fmt='%f')

