import numpy as np
from scipy.optimize import linprog
import pandas as pd

def calculate_F(bodies, contacts):
    print(f"Bodies: {bodies}")
    coefficients = np.zeros((3 * len(bodies), len(contacts) * 2 + len(bodies)), dtype=float)

    for n_body, body in enumerate(bodies):
        mass, cmx, cmy = body

        for n_contact, contact in enumerate(contacts):
            b1, b2, px, py, theta, mu = contact
            if b1 == n_body + 1 or b2 == n_body + 1:

                #friction cone angle
                alpha = np.arctan(mu)

                #set direction
                if b2 == n_body + 1 and b1 != 0:
                    direction = -1
                else:
                    direction = 1

                #unit vectors of friction cone edges
                normal1 = [np.cos(theta + alpha)*direction, np.sin(theta + alpha)*direction]
                normal2 = [np.cos(theta - alpha) * direction, np.sin(theta - alpha) * direction]

                #moment magnitudes
                m1 = np.cross([px, py], normal1)
                m2 = np.cross([px, py], normal2)

                #wrenches
                Fa = [normal1[0], normal1[1], float(m1)]
                Fb = [normal2[0], normal2[1], float(m2)]

                new_wrenches = np.array([Fa, Fb]).T

                coefficients[n_body * 3:n_body * 3 + new_wrenches.shape[0],
                             n_contact * 2:n_contact * 2 + new_wrenches.shape[1]] = new_wrenches

        fg = - mass * 9.81
        mg = np.cross([cmx, cmy], [0, fg])
        gravitational_wrench = np.array([0, fg, mg])
        coefficients[n_body * 3:n_body * 3 + gravitational_wrench.shape[0],
                     len(contacts) * 2 + n_body] = gravitational_wrench.T

    return coefficients

def assembly_stability(bodies, contacts):

    #calculate F
    F = calculate_F(bodies, contacts)
    rows, cols = len(F), len(F[0])

    #solve
    f = np.ones(cols)
    A = - np.identity(cols)
    b = -np.ones(cols)
    Aeq = F
    beq = np.zeros(rows)

    res = linprog(c=f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='highs')

    if res.success:
        print('Assembly stands')
        np.set_printoptions(precision=3, suppress=True)
        print(f"k: {res.x[:-2]}\n")
    else:
        print("Assembly collapses\nNo solution\n")

if __name__ == '__main__':

    # Ex 1: two leaning bodies assembly
    leaning_collapsing_bodies = [(2, 25, 35), (5, 66, 42)]
    leaning_standing_bodies = [(2, 25, 35), (10, 66, 42)]

    leaning_collapsing_contacts = [(0, 1, 0, 0, np.pi / 2, 0.1), (1, 2, 60, 60, np.pi, 0.5),
                                   (0, 2, 60, 0, np.pi / 2, 0.5), (0, 2, 72, 0, np.pi / 2, 0.5)]
    leaning_standing_contacts = [(0, 1, 0, 0, np.pi / 2, 0.5), (1, 2, 60, 60, np.pi, 0.5),
                                 (0, 2, 60, 0, np.pi / 2, 0.5), (0, 2, 72, 0, np.pi / 2, 0.5)]

    print("----Ex 1: 2 bodies leaning assembly-----")
    print("Collapsing configuration: ")
    assembly_stability(leaning_collapsing_bodies, leaning_collapsing_contacts)

    print("Standing configuration: ")
    assembly_stability(leaning_standing_bodies, leaning_standing_contacts)

    # Ex 2: arch assembly
    mass = 2
    arch_bodies = [(mass, 41, 16.5), (mass, 79, 16.5), (mass, 50, 33)]

    mu = 0.1
    arch_collapsing_contacts = [(0, 1, 0, 0, np.pi / 2, mu), (0, 1, 23.5, 0, np.pi / 2, mu),
                                (1, 3, 24.5, 42.5, 5 / 4 * np.pi, mu), (1, 3, 37, 22.5, 5 / 4 * np.pi, mu),
                                (2, 3, 63.5, 22.5, 7 / 4 * np.pi, mu), (2, 3, 75.5, 42.5, 7 / 4 * np.pi, mu),
                                (0, 2, 76, 0, np.pi / 2, mu), (0, 2, 100, 0, np.pi / 2, mu)]

    mu = 0.3
    arch_standing_contacts = [(0, 1, 0, 0, np.pi / 2, mu), (0, 1, 23.5, 0, np.pi / 2, mu),
                              (1, 3, 24.5, 42.5, 5 / 4 * np.pi, mu), (1, 3, 37, 22.5, 5 / 4 * np.pi, mu),
                              (2, 3, 63.5, 22.5, 7 / 4 * np.pi, mu), (2, 3, 75.5, 42.5, 7 / 4 * np.pi, mu),
                              (0, 2, 76, 0, np.pi / 2, mu), (0, 2, 100, 0, np.pi / 2, mu)]

    print("----Ex 2: 3 bodies arch assembly-----")
    print("Collapsing configuration: ")
    assembly_stability(arch_bodies, arch_collapsing_contacts)

    print("Standing configuration: ")
    assembly_stability(arch_bodies, arch_standing_contacts)
