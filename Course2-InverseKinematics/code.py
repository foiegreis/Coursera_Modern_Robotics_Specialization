from modern_robotics import *
import numpy as np
import csv
import pandas as pd


def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot
     - Prints out a report for each iteration of the Newton-Raphson process
     - from iterates 0 to the final solution

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev

    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.

    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001

    Logging:
        Iteration 3:
        joint vector: 0.221, 0.375, 2.233, 1.414
        SE(3) end-effector config: ...
        error twist V_b: ...
        angular error magnitude ||omega_b||: 0.357
        linear error magnitude ||v_b||: 1.427
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
        File.csv
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    joint_vector_list = []

    # Iteration 0 --------------------------------------------------------------------
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                      thetalist)), T)))
    omega_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = omega_b_norm > eomg \
          or v_b_norm > ev

    joint_vector_list.append(np.array(thetalist))

    # Logging------------------------------------------------------------------------
    print(f"Iteration {i}:\n")
    print(f"joint vector:\n {thetalist}", sep="\n", end="\n\n")
    print(f"SE(3) end-effector config:\n {FKinBody(M, Blist, thetalist)}", sep="\n", end="\n\n")
    print(f"error twist V_b:\n {Vb}", sep="\n", end="\n\n")
    print(f"angular error magnitude ||omega_b||: {omega_b_norm}", sep="\n", end="\n\n")
    print(f"linear error magnitude ||v_b||: {v_b_norm}", sep="\n", end="\n\n")
    print("_"*70)


    while err and i < maxiterations:
        thetalist = np.round(thetalist \
                    + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                         thetalist)), Vb), 4)
        i = i + 1
        Vb \
        = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                       thetalist)), T)))
        omega_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])

        err = omega_b_norm > eomg \
              or v_b_norm > ev

        #Logging
        print(f"Iteration {i}:\n")
        print(f"joint vector:\n {thetalist}", sep="\n", end="\n\n")
        print(f"SE(3) end-effector config:\n {FKinBody(M, Blist, thetalist)}", sep="\n", end="\n\n")
        print(f"error twist V_b:\n {Vb}", sep="\n", end="\n\n")
        print(f"angular error magnitude ||omega_b||: {omega_b_norm}", sep="\n", end="\n\n")
        print(f"linear error magnitude ||v_b||: {v_b_norm}", sep="\n", end="\n\n")
        print("_" * 70)

        joint_vector_list.append(np.array(thetalist))

    if not err:
        print(f"The algorithm has converged in {i} iterations")
    else:
        print(f"The algorithm has exceeded the maximum number of iterations (20)")

    df_indexed = pd.DataFrame(list(zip(*joint_vector_list))).add_prefix('theta')
    df_indexed.to_csv('iterates_indexed.csv', index_label='iteration')

    df = pd.DataFrame(list(zip(*joint_vector_list)))
    df.to_csv('iterates.csv', index=False)

    print("Csv file 'iterates.csv' has been created")

    return (thetalist, not err)

