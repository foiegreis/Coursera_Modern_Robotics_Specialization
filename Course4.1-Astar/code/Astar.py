import numpy as np
import pandas as pd
from tabulate import tabulate
from queue import PriorityQueue
from heapdict import heapdict
import time

from graph import Graph

class Astar:
    '''
    Evaluates Astar Algorithm
    '''
    def __init__(self, nodes_file, edges_file):
        '''
        Init
        :param nodes_file: csv file, format: x, y, heuristic-cost-to-go
        :param edges_file: csv file, format: ID1, ID2, cost
        :param solution_file: csv file, format: [solution path]
        '''
        self.time_start = time.time()
        self.Nodes = pd.read_csv(nodes_file, comment='#', sep=',', index_col=0)
        self.Edges = pd.read_csv(edges_file, comment='#', sep=',')

        print(f"\nEDGES\n {self.Edges}")
        print(f"\nNODES\n {self.Nodes}")

        #create graph
        g = Graph(self.Edges)
        g.print_graph()
        self.graph = g.get_graph()

        #goal node
        self.goal_node = self.Nodes.loc[self.Nodes['heuristic-cost-to-go'] == 0.].index.tolist()[0]
        print("GOAL NODE: ", self.goal_node)

        #initialize algorithm
        self.iter_count = 0
        self.optimal_path = []
        self.current = None
        self.n = len(self.Nodes.index)

        self.nodes_list = list(self.Nodes.index)
        print(self.nodes_list)
        self.past_cost = np.array([np.inf] * (self.n))

        self.open = heapdict()
        self.closed = []


        self.initialization()
        self.run_astar()

    def print_tabula(self):
        '''
        Prints tabulation of nodes and past cost, est_total_cost, parent_list
        :return:
        '''
        headers = list((range(1, self.n + 1)))

        m = [self.past_cost, self.opt_ctg, self.est_total_cost, self.parent_list]
        table = tabulate(m, headers, tablefmt='fancy_grid')
        print(table)

    def initialization(self):
        '''
        Initializes the algorithm with the start node
        :return:
        '''
        self.past_cost[0] = 0
        self.opt_ctg = np.array((self.Nodes.loc[:, 'heuristic-cost-to-go']))
        self.est_total_cost = np.add(self.past_cost, self.opt_ctg)
        self.parent_list = np.array([0]*self.n)

        self.update_open(self.nodes_list[0], self.opt_ctg[0])

        print("OPEN: ", list(self.open.items()))
        self.current = self.get_next_open()
        print(self.current)
        self.print_tabula()

    def recursive_cost_from_start(self, node_name):
        '''
        Calculates the cost from start to a node, recursively.
        The method will be used for updating the past cost when conditions meet
        :param node_name: node name
        :return: int
        '''
        if node_name == 1:
            return 0

        node_idx = node_name - 1
        parent = self.parent_list[node_idx]
        if parent != node_name:
            cost_from_parent = [x for x in self.graph[parent] if x[0]==node_name][0][1]
            return cost_from_parent + self.recursive_cost_from_start(parent)
        else:
            return self.past_cost[node_idx]

    def update_open(self, node, priority):
        '''
        Updates OPEN nodes heap
        :param node: node name
        :param priority: cost
        :return:
        '''
        #(node, est_total_cost)
        if node not in list(self.open.keys()):
            self.open[node] = priority
        else:
            #update if already present and priority < old priority
            if priority < self.open[node]:
                self.open[node] = priority

    def get_next_open(self):
        '''
        Gets current head of heap
        :return:
        '''
        next = self.open.peekitem()
        print(f"Next to explore: {next}")
        self.open.popitem()
        return next

    def update_closed(self, node):
        '''
        Updates CLOSED nodes list
        :param node: node name
        :return:
        '''
        self.closed.append(node)

    def explore(self):
        '''
        Explores the nodes using the A* logic
        :return:
        '''
        self.iter_count +=1
        parent = self.current[0]
        parent_idx = self.current[0]-1

        if parent not in self.closed:
            self.update_closed(parent)

            if parent != self.goal_node:
                parent_est_total_cost = self.current[1]
                print(f"\n***EXPLORING FROM NODE {parent} with est total cost {parent_est_total_cost}")
                cost_parent_from_start = self.recursive_cost_from_start(parent)
                #print("PARENT PAST COST: ", cost_parent_from_start)

                children = self.graph[parent]
                #print(f'children of node {parent}: {children}')

                for child in children:
                    if child[0] not in self.closed:
                        child_name = child[0]
                        child_idx = child_name-1
                        #print(f"modifying child {child_name}")
                        cost = child[1]

                        if self.parent_list[parent_idx] != 0:
                            cumulative_cost_from_start = cost + cost_parent_from_start #20
                            #print(f"CUMULATIVE COST FROM START {cost} + {cost_parent_from_start} = {cumulative_cost_from_start}")
                            #print(f"CURRENT PAST COST {self.past_cost[child_idx]}")

                            if cumulative_cost_from_start < self.past_cost[child_idx]:
                                self.past_cost[child_idx] = cumulative_cost_from_start
                                self.parent_list[child_idx] = parent

                        else:
                            cumulative_cost_from_start = cost
                            if cumulative_cost_from_start < cost:
                                self.past_cost[child_idx] = cumulative_cost_from_start
                            else:
                                self.past_cost[child_idx] = cost
                                self.parent_list[child_idx] = parent

                        #print("past_cost updated: ", self.past_cost[child_idx])
                        #print("parent node updated: ", self.parent_list[child_idx])
                        self.est_total_cost[child_idx] = np.add(self.past_cost[child_idx], self.opt_ctg[child_idx])
                        self.update_open(child_name, self.est_total_cost[child_idx])
                    else:
                        continue
            else:
                return
        print(f"Open nodes: {list(self.open.items())}")
        print(f"Closed nodes: {self.closed}")
        self.print_tabula()
        if self.open != []:
            self.current = self.get_next_open()
        else:
            return

    def recursive_evaluation(self, start, out):
        '''
        Gets the previous node parent from a start node (goal node)
        :param start: goal node
        :param out: empty list
        :return: list
        '''
        if start == 1:
            return out
        p = self.parent_list[start-1]
        out.append(p)
        return self.recursive_evaluation(p, out)

    def evaluate_result(self):
        '''
        Get True and evaluate optimal path if exists or False if no solution
        :return: bool
        '''
        if np.inf in self.past_cost:
            print("A* algorithm has no solutions")
            return False

        goal_idx = self.closed.index(self.goal_node)
        self.closed = self.closed[:goal_idx+1]
        out = [self.goal_node]
        self.optimal_path = self.recursive_evaluation(self.goal_node, out)[::-1]
        return True

    def run_astar(self):
        '''
        Runs algorithm
        '''
        for j in range(self.n):
            self.explore()

        print("*" * 60)
        print(f"The A* algorithm has terminated in {self.iter_count} steps")
        result = self.evaluate_result()
        print(f"RESULT: {result}\n")
        if result:
            time_end = time.time()
            elapsed_time = time_end - self.time_start
            print(f"A* algorithm has found solution in {round(elapsed_time, 3)} seconds")
            print(f"Optimal path: {self.optimal_path}")
        print("*" * 60)








