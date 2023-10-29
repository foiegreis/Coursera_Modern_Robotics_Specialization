import networkx as nx
import matplotlib.pyplot as plt
class Graph:
    '''
    Creates graph from pd.dataframe data, UNDIRECTED WEIGHTED GRAPH

    '''
    def __init__(self, data):
        '''
        Creates graph
        :param data: pd.dataframe [src, dst, weight]
        :return: dict: [src]: [(dst1, weight1), (dst2, weight2)]
        '''
        n = len(data)
        self.adj_dict = {}

        for j in range(n):
            vtx1, vtx2, weight = list(data.iloc[j, :])
            self.add_vertex(int(vtx1), int(vtx2))
            self.add_edge(int(vtx1), int(vtx2), weight)

        self.get_graph()

    def add_vertex(self, vtx1, vtx2):
        if vtx1 not in self.adj_dict:
            self.adj_dict[vtx1] = []
        if vtx2 not in self.adj_dict:
            self.adj_dict[vtx2] = []

    def add_edge(self, vtx1, vtx2, weight=0):
        if (vtx1, weight) not in self.adj_dict[vtx2]:
            self.adj_dict[vtx2].append((vtx1, weight))
        if (vtx2, weight) not in self.adj_dict[vtx1]:
            self.adj_dict[vtx1].append((vtx2, weight))

    def print_graph(self):
        print("\nGRAPH:\n")
        for vertex in self.adj_dict:
            print(vertex, ': ', self.adj_dict[vertex])

    def get_graph(self):
        return self.adj_dict

    def show_graph(self):
        G = nx.Graph()
        for node in self.adj_dict.keys():
            edges = self.adj_dict[node]
            if edges != []:
                for edge in edges:
                    G.add_edge(node, edge[0], weight=edge[1])

        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.1]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.1]
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=200)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()



