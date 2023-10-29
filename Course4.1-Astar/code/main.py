from Astar import Astar
from graph import Graph
import os
import pandas as pd

if __name__=='__main__':

    path_to_files = r"/Users/foiegreis/Desktop/code/ModernRobotics/Course4-Assignment1/solution/"
    nodes_file = os.path.join(path_to_files, 'nodes.csv')
    edges_file = os.path.join(path_to_files, 'edges.csv')

    algorithm = Astar(nodes_file, edges_file)
