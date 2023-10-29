'''
Copyrights: foiegreis 2023

Code for Modern Robotics Coursera - Course 5 - Assessment 1

Implementation of Linear programming test for First-Order form Closure

F = [F1, F2, ..., Fj] € R(nxj),
Fi = contact wrench = [miz, fix, fix].T

The contacts yield form closure if there exists a vector of weights k € R(j), k >=0, such that:
    - rank F is full
    - exist k > 0 such that Fk = 0

Algorithm:
    - find k
    - minimizing 1T*k
    - such that Fk = 0 ki >= 1, i = 1, ..., j

Linprog:
    - objective function as vector of weights f on the elements of k
    - A = matrix, -1 on diagonal
    - b = vector of -1
    - so Ak <= b
    - inequality constraints on k in the form Ak <= b
    - inequality constraints of the form AeqK = beq

return: None if no form closure, vector of k values if form closure

'''

import numpy as np
from scipy.optimize import linprog

def form_closure(contact_points, normal):

    n = 3 #planar rigid bodies

    print(f"Contact points\n{contact_points}\n")
    print(f"Normals\n{n_directions}\n")

    F = np.zeros((n, normal.shape[0]))

    #F matrix
    for i in range(len(contact_points)):
        norm = np.array([np.cos(normal[i]), np.sin(normal[i]), 0])
        cross = np.cross(contact_points[i], norm)
        F[:,i] = [cross[2], norm[0], norm[1]] # [wz, vx, vy]

    print(f"F matrix\n{F}\n")


    f = np.ones(len(F[0]))
    A = np.zeros((F.shape[1], F.shape[1]))
    np.fill_diagonal(A, -np.ones(F.shape[1]))
    b = -np.ones(len(F[0]))

    Aeq = F
    beq = np.zeros(F.shape[0])
    res = linprog(f, A, b, Aeq, beq, method='highs')
    success = res.success
    k = res.x
    print("Solution:\n")
    print(f"Form closure: {success} with k= {k}\n")
    return success, k

if __name__ == '__main__':

    #NO FORM CLOSURE
    contact_points = np.array([[0, 1, 0], [1, 1, 0]])
    n_directions = np.array([0, 5/4*np.pi])
    print("---Case1---")
    fc, k1 = form_closure(contact_points, n_directions)

    #FORM CLOSURE
    contact_points2 = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
    n_directions2 = np.array([0, 5/4*np.pi, np.pi/2])
    print("---Case2---")
    k2 = form_closure(contact_points2, n_directions2)

