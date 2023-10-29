import os
from RRT import RRT

if __name__=='__main__':
    path_to_files = r"/Users/foiegreis/Desktop/code/ModernRobotics/Course4.2-PRM_RRT/solution"
    obstacles_file = os.path.join(path_to_files, 'obstacles.csv')

    RRT_solution = RRT(obstacles_file)
