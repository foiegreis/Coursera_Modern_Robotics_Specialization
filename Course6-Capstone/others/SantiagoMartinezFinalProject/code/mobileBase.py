import numpy as np
from utils import SE3, SO3
import math

class MobileBase:

    # defining base characteristics
    l = 0.235 # (m) half distance between forward and back wheels
    w = 0.15 # (m) half distance between sides of the base 
    r = 0.0475 # (m) radius of each wheel
    h = 0.0963 # (m) height of body frame

    # defining the F matrix. see eqn 13.33 in Modern Robotics Textbook
    F = np.array([[-r/(4*(l+w)), r/(4*(l+w)), r/(4*(l+w)), -r/(4*(l+w))], 
                [r/4, r/4, r/4, r/4], 
                [-r/4, r/4, -r/4, r/4]])


    # defining the mecanum wheel "free sliding" directions
    gamma_1 = -np.pi*0.25 
    gamma_2 = np.pi*0.25
    gamma_3 = -np.pi*0.25 
    gamma_4 = np.pi*0.25 

    prevWheelPositions = np.zeros((4,1))

    def __init__(self, initialWheelPositions = None, initialChassisConfig = np.array([]), initialPlanarTwist = None):
        """
        Constructor for the mobileBase of the youBot. Sets the initial values for the wheel positions,
        the chassis configuration, and the planar twist.

        The wheel numbering is: (1) front left, (2) front right, (3) back right, (4) back left.
        
        The chassis configuration is: (phi) chassis angle, (x, y) position of the chassis frame with respect 
        to the global origin

        The body twist is: (omega_b_z) angular velocity about the z-axis of the chassis frame, (niu_b_x) chassis 
        velocity in x-axis of chassis frame, (niu_b_y) chassis velocity in y-axis of chassis frame

        Parameters
        ----------
        initialWheelPositions: a 4x1 2D numpy array containing the initial wheel positions
        initialChassisConfig: a 3x1 2D numpy array containing the initial chassis configuration
        initialPlanarTwist: a 3x1 2D numpy array containing the initial planar twist. 

        Returns
        -------
        None
        """
        if initialWheelPositions == None:
            self.wheelPositions = np.zeros((4,1))
        else:
            self.wheelPositions = np.copy(initialWheelPositions)
        
        if np.size(initialChassisConfig) == 0:
            self.q = np.zeros((3,1))
        else:
            self.q = np.copy(initialChassisConfig)
        
        if initialPlanarTwist == None:
            self.niu_b = np.zeros((6,1))
        else:
            self.niu_b = np.array([[0], [0], [initialPlanarTwist[0, 0]], [initialPlanarTwist[1, 0]], [initialPlanarTwist[2, 0]], [0]])

    
    def configToSE3(self) -> SE3:
        """
        Convert the current configuration of the mobileBase to an SE3 representation

        Parameters
        ---------
        
        Returns
        -------
        SE3 - 4x4 numpy array
        """

        phi = self.q[0, 0] # (rad) angle between chassis frame and global frame
        x = self.q[1, 0] # (m) horizontal position of the chassis frame and global frame
        y = self.q[2, 0] # (m) vertical position of the chassis frame and global frame
        R = SO3.rotationAboutZ(phi)
        r = np.array([[x], [y], [self.h]])

        T = SE3.toSE3(R,r)
        return T
    
    def step(self, ctrl, dt):
        """
        Parameters
        ---------
        ctrl: A 4x1 numpy array of wheel velocities
        dt: the simulation timestep

        Returns
        ---------

        """
        self.prevWheelPositions = np.copy(self.wheelPositions) # update last wheel configuration
        self.wheelPositions = np.add(self.wheelPositions, ctrl*dt) # update wheel configuration
        self.odometer() # update chassis configuration


    def odometer(self) -> None:
        '''
        Using newly measured wheel positions, calculate and update the configuration of the mobileBase
        
        Parameters
        ----------

        Returns
        -------
        None
        '''

        # implementing eqn 13.33 niu_b = F*theta_dot
        delta_q_b = np.zeros_like(self.q)

        theta_dot = np.subtract(self.wheelPositions, self.prevWheelPositions)
        new_niu_b = np.dot(self.F, theta_dot)

        # implementing eqn 13.35
        if math.isclose(new_niu_b[0], 0.0):
            delta_q_b[0] = new_niu_b[0]
            delta_q_b[1] = new_niu_b[1]
            delta_q_b[2] = new_niu_b[2]
        else:
            delta_q_b[0] = new_niu_b[0]
            delta_q_b[1] = (new_niu_b[1]*np.sin(new_niu_b[0]) + new_niu_b[2]*(np.cos(new_niu_b[0]) - 1))/new_niu_b[0]
            delta_q_b[2] = (new_niu_b[2]*np.sin(new_niu_b[0]) + new_niu_b[1]*(1 - np.cos(new_niu_b[0])))/new_niu_b[0]

        # implementing eqn 13.36
        phi = self.q[0,0]
        delta_q_s = np.dot(SO3.rotationAboutX(phi), delta_q_b)
        
        # updating configuration
        self.q = np.add(self.q, delta_q_s) # eqn 13.36


        
        




