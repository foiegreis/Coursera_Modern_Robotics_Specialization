import numpy as np
import modern_robotics as mr



if __name__=='__main__':

    #course 5 - final quiz - Q6
    #Je = [Jbase, Jarm]
    #Ve = Je * u

    #Jbase
    Tbe = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 0], [0, 0, 0, 1]])
    adj = np.linalg.inv(mr.Adjoint(Tbe))     #[Ad(tbe-1)] = Ad(tbe)^-1
    F = np.array([[0, 0], [0, 0], [-0.25, 0.25], [0.25, 0.25], [0, 0], [0, 0]])
    Jbase = np.dot(adj, F)
    print("JBASE \n", Jbase)

    #Jarm
    w1 = np.array([0, 0, 1])
    p1 = np.array([-3, 0, 0])
    v1 = - np.cross(w1, p1)
    B = np.concatenate([w1, v1])
    thetalist = np.array([np.pi/2])
    Blist = np.array([B])
    Jarm = mr.JacobianBody(Blist, thetalist)
    print('\nJARM \n', Jarm)