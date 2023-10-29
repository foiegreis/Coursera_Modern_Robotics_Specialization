import math
import numpy as np

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    def __repr__(self):
        return f"(Node: x={self.x}, y={self.y}, parent={self.parent})"
class Tree:
    def __init__(self):
        ''' Weighted undirected graph '''
        self.nodes = []
        self.tree = {0: []}
        self.edges = []

    def __repr__(self):
        return f"Tree: {self.tree}"

    @staticmethod
    def dist(p1, p2):
        x, y = p1
        xx, yy = p2
        return np.round(math.hypot(x - xx, y - yy), 4)

    def add_node(self, pos, parent=None):
        id = len(self.nodes)
        node = Node(*pos, parent)
        self.nodes.append(node)
        return id

    def add_edge(self, parent_id, node_id, cost):
        try:
            self.tree[node_id].append((parent_id, cost))
        except:
            self.tree[node_id] = [(parent_id, cost)]

    def get_coords(self, id):
        node = self.nodes[id]
        return [node.x, node.y]

    def find_nearest(self, sample):
        distances = []
        for n in self.nodes:
            x, y, parent = self.nodes[n]
            d = self.dist((x, y), sample)
            distances.append((n, d))
        return min(distances, key=lambda x: x[1])





