Copyrights Greta Russi, 2023
Implementation of the RRT algorithm for path planning

You may find configurations in config.ini:
[environment]
n = 2
limits_left = -0.5
limits_right = 0.5

[robot]
robot_radius = 0.02

[rrt]
n_samples = 400
goal_percent = 0.05

[run]
max_iter = 500
start = (-0.5, -0.5)
goal = (0.5, 0.5)
goal_rad = 0.04
step_size = 0.1

Run:
python3 main.py

Output:
solution/edges.csv, solution/path.csv, solution/nodes.csv
solution/tree.png, solution/path.png
