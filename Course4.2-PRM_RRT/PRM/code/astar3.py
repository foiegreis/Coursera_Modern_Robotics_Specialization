import numpy as np


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


remove_comments("edges.csv", "edges2.csv")
edges_file = open("edges2.csv")
edges_data = np.loadtxt(edges_file, delimiter=",")

remove_comments("nodes.csv", "nodes2.csv")
nodes_file = open("nodes2.csv")
nodes_data = np.loadtxt(nodes_file, delimiter=",")

optimist_ctg = nodes_data.T[3]
past_cost = np.full(len(nodes_data), 1e6)
past_cost[0] = 0
est_tot_cost = optimist_ctg + past_cost
parent_node = np.full(len(nodes_data), -1, dtype=int)

OPEN = [1]
CLOSED = []
path = []
N = len(optimist_ctg)
print(N)
last_node = N

sort_matrix = np.full((N, 2), 1e6, dtype=float)

while len(OPEN) > 0:
    current = OPEN[0]
    OPEN.pop(0)  # remove from OPEN
    CLOSED.append(current)  # add current to CLOSED
    if current == last_node:    # SUCCESS
        path.append(current)
        index = int(current)
        while index > 1:
            path.append(parent_node[index-1])
            index = parent_node[index-1]
        path.reverse()
        for number in range(len(path)):
            path[number] = int(path[number])
        print("SUCCESS")
        print(path)
        spath = str(path)
        spath = spath.replace("[", "").replace("]", "")  # remove brackets
        with open("path.csv", "w") as fp:
            fp.write(spath)
        break
    for number in range(len(edges_data)):   # look for children of current
        if edges_data[number][1] == current:
            nbr = int(edges_data[number][0])  # child node found
            if nbr in CLOSED:
                continue
            currentm1 : int = int(current - 1)
            tentative_past_cost = past_cost[currentm1] + edges_data[number][2]
            if tentative_past_cost < past_cost[nbr - 1]:
                past_cost[nbr-1] = tentative_past_cost
                parent_node[nbr-1] = current
                if nbr not in OPEN:
                    OPEN.append(nbr)
                est_tot_cost[nbr-1] = past_cost[nbr-1] + optimist_ctg[nbr-1]

                # sort OPEN
                for number in range(N):
                    sort_matrix[number][0] = 0
                    sort_matrix[number][1] = 1e6
                for number in range(len(OPEN)):
                    sort_matrix[number][0] = OPEN[number]
                    index: int = int(OPEN[number] - 1)
                    sort_matrix[number][1] = est_tot_cost[index]
                sort_matrix = sort_matrix[np.argsort(sort_matrix[:,1])]
                for number in range(len(OPEN)):
                    OPEN[number] = sort_matrix[number][0]

print("ENDED")





