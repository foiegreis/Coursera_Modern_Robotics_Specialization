from youBot import youBot
from utils import SE3, SO3
from executiveControl import ExecutiveControl
import numpy as np

def main():
    # Velocity control Limits = [u1_lim u2_lim u3_lim u4_lim th_dot1_lim th_dot2_lim th_dot3_lim th_dot4_lim th_dot5_lim]
    velocityControlLimits = [15, 15, 15, 15,   2, 2, 6, 6, 6]

    # Velocity Control Input = [u1, u2, u3, u4, th_dot1, th_dot2, th_dot3, th_dot4, th_dot5] 
    velocityControlInput = [10, 10, 10, 10,    3, -3, 0.1, -0.6, 0.8]
    
    # cube initial position [1 0 0 1; 0 1 0 0; 0 0 1 0.025; 0 0 0 1]
    T_s_c_i = SE3.toSE3(SO3.I(), np.array([[1], [0], [0.025]]))

    # cube final position [0 1 0 0; -1 0 0 -1; 0 0 1 0.025; 0 0 0 1]
    T_s_c_f = SE3.toSE3(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), np.array([[0], [-1], [0.025]]))

    executiveController = ExecutiveControl(standoffDistance=0.2, youBot=youBot(wheelPositions=None, chassisConfig=np.array([[0], [0], [0]]), planarTwist=None, jointConfig=np.array([[0],[0],[0],[0],[0]])))
    
    executiveController.Simulate(T_s_c_i, T_s_c_f, velocityControlLimits)

if __name__ == "__main__":
    main()
    