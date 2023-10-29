import modern_robotics as mr
import numpy as np
import scipy

#calculate inverse kinematics theta = (th1, th2, th3) that put 3R end effector to Tsd
Tsd = np.array([[-0.585, -0.811, 0, 0.076],
               [0.811, -0.585, 0, 2.608],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

thetalist0 = np.array([np.pi/4, np.pi/4, np.pi/4])
eps_w = 0.001
eps_v = 0.0001

Blist = np.array([[0, 0, 1, 0, 3, 0],
                  [0, 0, 1, 0, 2, 0],
                  [0, 0, 1, 0, 1, 0]]).T

M = np.array([[1, 0, 0, 3],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

[thetalist, success] = mr.IKinBody(Blist, M, Tsd, thetalist0, eps_w, eps_v)
print("thetalist: ", thetalist)
print("success: ", success)


