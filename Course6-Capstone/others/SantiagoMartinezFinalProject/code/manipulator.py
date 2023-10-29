import numpy as np
import modern_robotics as mr
from utils import SE3, SO3

class Manipulator:
    # defining the manipulator characteristics
    # M_0_e = [1 0 0 0.033; 
    #          0 1 0 0;
    #           0 0 1 0.6546;
    #            0 0 0 1]
    M_0_e = SE3.toSE3(SO3.I(), np.array([[0.033], [0], [0.6546]]))

    # B_i = (omg_i, niu_i) 
    # each column is a screw, notice the transpose at the end
    Blist = np.array([[0, 0, 1,   0, 0.033, 0],
                     [0, -1, 0, -0.5076, 0, 0],
                     [0, -1, 0, -0.3526, 0, 0],
                     [0, -1, 0, -0.2176, 0, 0],
                     [0,  0, 1,       0, 0, 0]]).T
    
    jointAngles = np.zeros((5,1))
    
    def __init__(self, initialConfig=np.array([])):
        """
        Constructor for the manipulator of the youBot. 

        Parameters
        ----------
        initialConfig: a 5x1 numpy array containing the initial joint configuration. 

        Returns
        -------
        None
        """
        if np.size(initialConfig) == 0:
            self.updateConfig(np.zeros((5,1)))
            self.T_0_e = self.M_0_e
        else:
            self.updateConfig(initialConfig)
           
    def updateConfig(self, newConfig):
        """
        Updates the configuration of the manipulator with the given angles

        Parameters
        ----------
        newConfig: a 5x1 numpy array containing the new joint configuration. 

        Returns
        -------
        None
        """
        self.jointAngles = np.copy(newConfig)

        self.T_0_e = mr.FKinBody(self.M_0_e, self.Blist, self.jointAngles)

    def step(self, ctrl, dt):
        """
        Given velocity control inputs, update the configuration of the manipulator

        Parameters
        ----------
        ctrl: a 5x1 numpy array containing the joint velocities. 
        dt: the simulation timestep

        Returns
        -------
        None
        """
        self.updateConfig(np.add(self.jointAngles, ctrl*dt))

    
        
    