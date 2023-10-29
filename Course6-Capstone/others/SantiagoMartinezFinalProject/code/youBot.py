import numpy as np
from mobileBase import MobileBase
from manipulator import Manipulator
from utils import SE3, SO3, Output
import modern_robotics as mr

class youBot:
    # timestep for the sim is 0.01
    dt = 0.01
    t = 0
    
    # youBot Configuration = [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state]
    youBotInitConfig = [0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0,  0]
    state = youBotInitConfig

    # T_b_0 = [1 0 0 0.1662; 
    #          0 1 0 0;
    #           0 0 1 0.0026;
    #            0 0 0 1]
    # is the offset between the chassis frame {b} and the manipulator base {0}
    T_b_0 = SE3.toSE3(SO3.I(), np.array([[0.1662], [0], [0.0026]]))

    # initialize the csv output file
    csv = Output("./output/youBotState.csv", ["phi", "x", "y", "J1", "J2", "J3", "J4", "J5", "W1", "W2", "W3", "W4", "gripper"])
    
    def __init__(self, wheelPositions, chassisConfig, planarTwist = None, jointConfig = np.array([])):
        """
        Constructor of the youBot. Creates an instance for the MobileBase and the Manipulator, which compose the youBot.

        Parameters
        ----------
        wheelPositions: a 4x1 numpy array containing the initial wheel positions.
        chassisConfig: a 3x1 numpy array containing the initial chassis configuration.
        planarTwist: a 3x1 numpy array containing the initial planar twist. 
        jointConfig: a 5x1 numpy array containing the initial joint configuration. 

        Returns
        -------
        None
        """
        self.base = MobileBase(wheelPositions, chassisConfig, planarTwist)
        self.manipulator = Manipulator(jointConfig)

        self.F_6 = np.concatenate([np.zeros([1,4]), np.zeros([1,4]), self.base.F, np.zeros([1,4])], axis=0)
        T_s_b = self.base.configToSE3()
        T_b_0 = self.T_b_0
        T_0_e = self.manipulator.T_0_e

        # initial end effector configuration in the {s} frame
        self.T_s_e = SE3.multiplication(T_s_b, SE3.multiplication(T_b_0, T_0_e))

        # update body jacobian
        self.J_b_arm = mr.JacobianBody(self.manipulator.Blist, self.manipulator.jointAngles)
        self.J_b_base = np.matmul(mr.Adjoint(SE3.multiplication(mr.TransInv(T_0_e), mr.TransInv(T_b_0))), self.F_6)

        self.J_b = np.concatenate([self.J_b_base, self.J_b_arm], axis=1)

    def Step(self, velCtrlInput, velCtrlLimits=None, gripperInput=0, isWritten=False):
        """
        Steps the simulation by updating the wheel and joint configuration of the robot, given the control parameters

        Parameters:
        -----------
        velCtrlInput: a 1x9 list of the form [u1, u2, u3, u4, th_dot1, th_dot2, th_dot3, th_dot4, th_dot5]
        isWritten: flag to determine if the state is written to csv. 
        velCtrlLimits: a 1x9 list of limits of velocity control inputs for the wheels and joints [u1_lim u2_lim u3_lim u4_lim th_dot1_lim th_dot2_lim th_dot3_lim th_dot4_lim th_dot5_lim]

        Return:
        -----------
        None
        """
        self.t += self.dt 

        if isWritten:
            self.csv.AddState(self.state)

        if velCtrlLimits == None:
            ctrlArr = velCtrlInput
        else:
            # check if the ctrl inputs are beyond the limits
            ctrlArr = velCtrlInput
            idx = 0
            for ctrl in velCtrlInput:

                if abs(ctrl) > velCtrlLimits[idx]:
                    sign = ctrl/abs(ctrl)
                    ctrlArr[idx, 0] = sign*velCtrlLimits[idx]

                idx += 1
        
        self.base.step(ctrlArr[0:4], self.dt)
        self.manipulator.step(ctrlArr[4:9], self.dt)

        self.UpdateState(gripperInput)
      
    def UpdateState(self, gripperInput = 0):
        """
        Aggregates the base and manipulator states into one.

        Parameters:
        -----------
        None 

        Return:
        -----------
        None
        """
        self.state = []
        for i in range(np.shape(self.base.q)[0]):
            self.state.append(self.base.q[i,0])
        for i in range(np.shape(self.manipulator.jointAngles)[0]):
            self.state.append(self.manipulator.jointAngles[i,0])
        for i in range(np.shape(self.base.wheelPositions)[0]):
            self.state.append(self.base.wheelPositions[i,0])
        # append gripper
        self.state.append(gripperInput)

        # perform forward kinematics to obtain the EE config
        T_s_b = self.base.configToSE3()
        T_b_0 = self.T_b_0
        T_0_e = self.manipulator.T_0_e

        # end effector configuration in the {s} frame
        self.T_s_e = SE3.multiplication(T_s_b, SE3.multiplication(T_b_0, T_0_e))

        # update body jacobian
        self.J_b_arm = mr.JacobianBody(self.manipulator.Blist, self.manipulator.jointAngles)
        self.J_b_base = np.matmul(mr.Adjoint(SE3.multiplication(mr.TransInv(T_0_e), mr.TransInv(T_b_0))), self.F_6)

        self.J_b = np.concatenate([self.J_b_base, self.J_b_arm], axis=1)
        



