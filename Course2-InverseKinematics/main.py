"""
Course 2 - Peergrad assignment
UR5 robot inverse kinematics - Implementation of Newton-Raphson function from Modern Robotics's IKinBody
__author__ = "foiegreis"
"""

from code import IKinBodyIterates
import numpy as np

#UR5 Robot -----------------------------------------------------
Blist = np.array([[0, 1, 0, 0.191, 0, 0.817],
                  [0, 0, 1, 0.095, -0.817, 0],
                  [0, 0, 1, 0.095, -0.392, 0],
                  [0, 0, 1, 0.095, 0, 0],
                  [0, -1, 0, -0.082, 0, 0],
                  [0, 0, 1, 0, 0, 0]]).T

M = np.array([[-1, 0, 0, 0.817],
              [0, 0, 1, 0.191],
              [0, 1, 0, -0.006],
              [0, 0, 0, 1]])

T = np.array([[0, 1, 0, -0.5],
              [0, 0, -1, 0.1],
              [-1, 0, 0, 0.1],
              [0, 0, 0, 1]])

thetalist0 = np.array([2.5, 0.7, 4.1, -2.3, 0.46, -3.6])
eomg = 0.001
ev = 0.0001

[thetalist, success] = IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
print("_"*70)
print("thetalist:", thetalist)
print("success:", success)