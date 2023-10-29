import matplotlib.pyplot as plt

def draw_assembly(assembly):
    plt.figure()
    for body in assembly.keys():
        coord = assembly[body]
        # repeat the first point to create a 'closed loop'
        coord.append(coord[0])
        xs, ys = zip(*coord)
        plt.plot(xs, ys)
    plt.show()

if __name__=='__main__':

    arch = {1: [(0, 0), (23.5, 0), (37, 22.5), (24.5, 42.5)],
                    2: [(24.5, 42.5), (37, 22.5), (63.5, 22.5), (75.5, 42.5)],
                    3: [(76, 0), (100, 0), (75.5, 42.5), (63.5, 22.5)]}


    draw_assembly(arch)
