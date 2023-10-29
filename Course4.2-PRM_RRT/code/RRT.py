'''
Copyrights: Greta Russi
'''
from configparser import ConfigParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import csv
import os

from tree import Tree


class RRT:
    def __init__(self, obstacles_file, test=False):
        self.test = test
        self.tree = None
        self.path_to_save = os.path.abspath('..')

        columns = ['x', 'y', 'diameter']
        self.obstacles = pd.read_csv(obstacles_file, comment='#', sep=',', header=None, names=columns)

        config = ConfigParser()
        config.read('config.ini')

        self.n = eval(config.get('environment', 'n'))
        self.limits_left = eval(config.get('environment', 'limits_left')) ##
        self.limits_right = eval(config.get('environment', 'limits_right'))
        self.robot_radius = eval(config.get('robot', 'robot_radius'))
        self.n_samples = eval(config.get('rrt', 'n_samples'))
        self.goal_percent = eval(config.get('rrt', 'goal_percent'))
        self.max_iter = eval(config.get('run', 'max_iter'))
        self.start = eval(config.get('run', 'start'))
        self.goal = eval(config.get('run', 'goal'))
        self.goal_rad = eval(config.get('run', 'goal_rad'))
        self.step_size = eval(config.get('run', 'step_size'))

        self.rand_nodes = self.create_samples()
        self.run_RRT()

    @staticmethod
    def dist(p1, p2):
        # or: np.linalg.norm(p1-p2)
        return np.round(math.hypot(p1[0] - p2[0], p1[1] - p2[1]), 4)

    @staticmethod
    def sign(x):
        return -1 if x < 0 else 1

    def create_samples(self):
        ''' Random samples from environment, a percentage of them will be goal'''
        rand_nodes = np.random.random((self.n_samples, self.n)) - self.limits_right
        indices = np.random.choice(np.arange(len(rand_nodes)), replace=False,
                                   size=int(len(rand_nodes) * self.goal_percent))

        rand_nodes[indices] = self.goal
        rand_nodes[0] = self.start
        rand_nodes[-1] = self.goal
        return rand_nodes

    def collision_check(self, p1, p2, obst):
        ''' Checks if line between points 1 and 2 intersect obstacle - Code from Circle-LineIntersection'''
        xc, yc, diameter = obst
        r = diameter/2 + self.robot_radius
        x1 = p1[0] - xc
        y1 = p1[1] - yc
        x2 = p2[0] - xc
        y2 = p2[1] - yc
        dx = x2 - x1
        dy = y2 - y1
        dr = math.sqrt(dx * dx + dy * dy)
        D = x1 * y2 - x2 * y1
        delta = r * r * dr * dr - D * D

        if delta < 0:
            return False

        if delta == 0:
            xa = (D * dy) / (dr * dr)
            ya = (-D * dx) / (dr * dr)
            ta = (xa - x1) * dx / dr + (ya - y1) * dy / dr
            xpt = [(xa + xc, ya + yc)] if 0 < ta < dr else []
            if xpt == []:
                return False
            else:
                return True

        xa = (D * dy + self.sign(dy) * dx * math.sqrt(delta)) / (dr * dr)
        ya = (-D * dx + abs(dy) * math.sqrt(delta)) / (dr * dr)
        ta = (xa - x1) * dx / dr + (ya - y1) * dy / dr
        xpt = [(xa + xc, ya + yc)] if 0 < ta < dr else []

        xb = (D * dy - self.sign(dy) * dx * math.sqrt(delta)) / (dr * dr)
        yb = (-D * dx - abs(dy) * math.sqrt(delta)) / (dr * dr)
        tb = (xb - x1) * dx / dr + (yb - y1) * dy / dr
        xpt += [(xb + xc, yb + yc)] if 0 < tb < dr else []
        if xpt == []:
            return False
        else:
            return True

    def nearest(self, node):
        distances = []
        for i, n in enumerate(self.tree.nodes):
            d = self.dist(node, (n.x, n.y))
            distances.append((i, d))
        nearest = min(distances, key=lambda x: x[1])
        return nearest

    def local_planner(self, nearest_pose, sample_pose):
        ''' Returns node in the direction of sample, step-size away, if d > step size'''
        d = self.dist(nearest_pose, sample_pose)
        if d > self.step_size:
            (xnear, ynear) = nearest_pose
            (xsample, ysample) = sample_pose
            (dx, dy) = (xsample - xnear, ysample - ynear)
            theta = math.atan2(dy, dx)

            x = xnear + self.step_size * math.cos(theta)
            y = ynear + self.step_size * math.sin(theta)
            print(f"NEW POSE: {x}, {y}")
            return [x, y]
        else:
            return sample_pose

    def get_path(self):
        path = []
        node = self.tree.nodes[-1]
        while node.parent is not None:
            parent = node.parent
            path.append(parent)
            node = self.tree.nodes[parent]
        return path[::-1]

    def save_output(self, optimal_path):
        with open(os.path.join(self.path_to_save,'solution/nodes.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(self.tree.nodes)):
                node_x, node_y = self.tree.get_coords(i)
                writer.writerow([i+1, node_x, node_y])

        with open(os.path.join(self.path_to_save,'solution/edges.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for key, item in self.tree.tree.items():
                child_id = key
                parents = item
                for parent in parents:
                    writer.writerow([child_id+1, parent[0]+1, parent[1]])

        with open(os.path.join(self.path_to_save,'solution/path.csv'), 'w', newline='') as f:
            path = [x+1 for x in optimal_path]
            writer = csv.writer(f, delimiter=',')
            writer.writerow(path)


    @staticmethod
    def plot_collision_check(pt1, pt2, obstacles):
        ''' Plots collision check between obstacle and line between the two points pt1 and pt2 '''
        fig, ax = plt.subplots()
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

        for idx, obst in obstacles.iterrows():
            circle = plt.Circle((obst[0], obst[1]), obst[2] / 2, color='b', fill=False)
            ax.add_patch(circle)

        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'ro-')
        ax.axis('equal')
        plt.title("Collision check")
        plt.show()

    def plot_tree(self, show_path=False, save_path=None, optimal_path=None):
        ''' Plots collision check between obstacle and line between the two points pt1 and pt2 '''
        fig, ax = plt.subplots()
        plt.xlim([-0.6, 0.6])
        plt.ylim([-0.6, 0.6])

        #draw obstacles
        for idx, obst in self.obstacles.iterrows():
            circle = plt.Circle((obst[0], obst[1]), obst[2] / 2, color='b', fill=False)
            ax.add_patch(circle)

        #draw goal radius
        circle_goal = plt.Circle((self.goal[0], self.goal[1]), self.goal_rad, color='g', fill=True, alpha=0.5)
        ax.add_patch(circle_goal)

        #draw samples
        for i, s in enumerate(self.rand_nodes):
            if i == 0 or i == len(self.rand_nodes)-1:
                plt.plot(s[0], s[1], 'b*' )
            else:
                plt.plot(s[0], s[1], marker='o', markerfacecolor='red', markersize=0.8)

        #plot tree
        for key, item in self.tree.tree.items():
            parent_id = key
            parent_x = self.tree.nodes[parent_id].x
            parent_y = self.tree.nodes[parent_id].y
            children = item
            for ch in children:
                ch_id = ch[0]
                ch_x = self.tree.nodes[ch_id].x
                ch_y = self.tree.nodes[ch_id].y
                plt.plot([parent_x, ch_x], [parent_y, ch_y], 'go-')

        #plot optimal path
        if show_path:
            for i in range(len(optimal_path)):
                p1 = self.start if i == 0 else self.tree.get_coords(optimal_path[i-1])
                p2 = self.goal if i == len(optimal_path)-1 else self.tree.get_coords(optimal_path[i])
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y*-')
                plt.text(p1[0], p1[1], str(optimal_path[i]))

        plt.text(self.goal[0]+0.02, self.goal[1]+0.02, 'GOAL')
        plt.text(self.start[0]-0.05, self.start[1]-0.05, 'START')


        if save_path:
            fig.savefig(save_path)

        #ax.axis('equal')
        plt.title("RRT")
        plt.show()


    def run_RRT(self):
        ''' Performs RRT loop '''
        free = True
        T = 0

        #initialization
        self.tree = Tree()
        self.tree.add_node(self.start)
        print(self.tree.nodes)
        print(self.tree)

        while T < self.max_iter:
                print("ITER ", T)
                #1. pick a random sample point from the space: sample
                sample_id = random.choice(range(len(self.rand_nodes)))
                sample_pos = self.rand_nodes[sample_id]

                #2. find nearest in tree to sample: near
                near_id, near_d = self.nearest(sample_pos)
                near_pos = self.tree.get_coords(near_id)

                #3. run local planner to generate: new
                new_pos = self.local_planner(near_pos, sample_pos)
                print("New node: ", new_pos)

                #4. check collision
                collisions = []
                for i, obst in self.obstacles.iterrows():
                    c = self.collision_check(new_pos, near_pos, obst)
                    collisions.append(c)

                free = not any(collisions)
                if free:
                    #5. add new to the tree
                    if (new_pos[0] != self.start[0]) and (new_pos[1] != self.start[1]):
                        new_id = self.tree.add_node(new_pos, parent=near_id)
                        dist = self.dist(near_pos, new_pos)

                        if dist != 0.:
                            self.tree.add_edge(near_id, new_id, dist)

                            #6. check if success
                            d = self.dist(new_pos, self.goal)
                            if d <= self.goal_rad:
                                print(f"SUCCESS in {T} STEPS")
                                self.plot_tree(save_path=os.path.join(self.path_to_save, 'solution/tree.png'))

                                #8. search optimal path
                                optimal_path = self.get_path()
                                optimal_path.append(new_id)
                                print("optimal path ", optimal_path)
                                self.plot_tree(show_path=True,
                                               save_path=os.path.join(self.path_to_save, 'solution/path.png'),
                                               optimal_path=optimal_path)

                                #7.save files
                                self.save_output(optimal_path)
                                return True

                T += 1
        self.plot_tree()
        return False



'''
if __name__=='__main__':
    path_to_files = r"/Users/foiegreis/Desktop/code/ModernRobotics/Course4.2-PRM_RRT/"
    obstacles_file = os.path.join(path_to_files, 'obstacles.csv')

    rrt = RRT(obstacles_file)
'''











