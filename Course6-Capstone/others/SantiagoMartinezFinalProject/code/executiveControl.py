import csv
import numpy as np
import modern_robotics as mr
import youBot
from utils import SE3, SO3, Output

class TrajectoryOutput:
    """
    Class used to write the trajectory to a csv
    """    
    def __init__(self, csvFileName, fieldNames):  
        self.csvTable = []
        self.csvFileName = csvFileName
        self.fieldNames = fieldNames

    def AddWaypoint(self, waypoint, grapple):
        wpDict = {}
        idx = 0
        for i in range(0,3):
            for j in range(0,3):
                name = self.fieldNames[idx]
                wpDict[name] = waypoint[i, j]
                idx += 1
        for i in range(0,3):
            name = self.fieldNames[idx]
            wpDict[name] = waypoint[i, 3]
            idx += 1

        name = self.fieldNames[idx]
        wpDict[name] = grapple

        self.csvTable.append(wpDict)

    def AddGrapplerWaypoint(self, grapple, dt):
        wp = self.csvTable[-1]
        for i in range(int(0.64/dt)):
            wp["gripper"] = grapple
            self.csvTable.append(wp)

    def WriteTrajectoryToCSV(self):
        with open(self.csvFileName, 'w') as csvfile:
            csvWriter = csv.DictWriter(csvfile, fieldnames=self.fieldNames)
            for entry in self.csvTable:
                csvWriter.writerow(entry)

    
class ExecutiveControl:

    csv = TrajectoryOutput("./output/endEffectorTraj.csv", ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "px", "py", "pz", "gripper"])
    errorCsv = Output("./output/trajError.csv", ["omgX", "omgY", "omgZ", "niuX", "niuY", "niuZ"])
    def __init__(self, standoffDistance, youBot):
        self.N = 10                         # The number of points N > 1 (Start and stop) in the discrete representation of the trajectory
        self.method = 5                     # The time-scaling method
        self.standoff = standoffDistance    # The distance from the structure to transition
        self.youBot = youBot
        self.maxLinV = 0.2                  # m/s
        self.maxAngV = 0.5                  # rad/s

        self.trajectory = []
        self.grapplerTraj = []

        self.error_prior = np.zeros([6, 1])                # is a twist
        self.integral_prior = np.zeros([6, 1])              # is a twist

    def TrajMoveTo(self, Xstart, Xend, grappler):    
        # linear distance between the two points
        iPos = SE3.getPosition(Xstart)
        fPos = SE3.getPosition(Xend)
        linD = np.linalg.norm(np.subtract(fPos, iPos))

        # angular distance between the two end points
        iRot = SE3.getRotation(Xstart)
        fRot = SE3.getRotation(Xend)
        R_start_end = SO3.multiplication(mr.RotInv(iRot),fRot)
        if (np.array_equal(R_start_end, SO3.I())):
            axis, angD  = ([0, 0, 0], 0)
        else:
            axis, angD = mr.AxisAng3(mr.so3ToVec(mr.MatrixLog3(R_start_end)))
        
        # max time of linear time and angular time
        tf = round(max(linD/self.maxLinV, angD/self.maxAngV),2)

        trajectory = mr.CartesianTrajectory(Xstart, Xend, tf, tf/self.youBot.dt, self.method)
        
        for wp in trajectory:
            self.csv.AddWaypoint(wp, grappler)
            self.grapplerTraj += [grappler]

        self.trajectory += trajectory
        

    def TrajGrapple(self):
        self.csv.AddGrapplerWaypoint(1, self.youBot.dt)
        wp = self.trajectory[-1]
        for i in range(int(0.64/self.youBot.dt)):
            self.trajectory.append(wp)
            self.grapplerTraj += [1]

    def TrajUngrapple(self):
        self.csv.AddGrapplerWaypoint(0, self.youBot.dt)
        wp = self.trajectory[-1]
        for i in range(int(0.64/self.youBot.dt)):
            self.trajectory.append(wp)
            self.grapplerTraj += [0]

    def GenerateTrajectory_MoveBlock(self, T_s_ci, T_s_cf):
        # Get initial T_s_e
        T_s_b = self.youBot.base.configToSE3()
        T_b_0 = self.youBot.T_b_0
        T_0_e = self.youBot.manipulator.M_0_e

        # initial end effector configuration in the {s} frame
        T_s_e0 = SE3.multiplication(T_s_b, SE3.multiplication(T_b_0, T_0_e))
      
        # end effector configuration above the block
        # some distance above the block, plus EE orientation facing the block
        T_ci_e1 = SE3.toSE3(SO3.rotationAboutY(np.pi), np.array([[0], [0], [self.standoff]])) 
        T_s_e1 = SE3.multiplication(T_s_ci, T_ci_e1)

        # move from initial state to hover position above the block, with grappler
        self.TrajMoveTo(T_s_e0, T_s_e1, 0)
        
        # move from hover to cube
        T_e1_e2 = SE3.toSE3(SO3.I(), np.array([[0], [0], [self.standoff]]))
        T_s_e2 = SE3.multiplication(T_s_e1, T_e1_e2)

        self.TrajMoveTo(T_s_e1, T_s_e2, 0)

        # grapple the block
        self.TrajGrapple()

        # move from cube to hover
        T_e2_e3 = SE3.toSE3(SO3.I(), np.array([[0], [0], [-self.standoff]]))
        T_s_e3 = SE3.multiplication(T_s_e2, T_e2_e3)

        self.TrajMoveTo(T_s_e2, T_s_e3, 1)

        # end effector configuration above the final pose cube
        # some distance above the block, plus EE orientation facing the block
        T_cf_e4 = SE3.toSE3(SO3.rotationAboutY(np.pi), np.array([[0], [0], [self.standoff]])) 
        T_s_e4 = SE3.multiplication(T_s_cf, T_cf_e4)

        # move from hover above initial cube to hover above final cube pose
        self.TrajMoveTo(T_s_e3, T_s_e4, 1)

        # move from hover to final cube pose
        T_e4_e5 = SE3.toSE3(SO3.I(), np.array([[0], [0], [self.standoff]]))
        T_s_e5 = SE3.multiplication(T_s_e4, T_e4_e5)

        self.TrajMoveTo(T_s_e4, T_s_e5, 1)

        # ubgrapple the block
        self.TrajUngrapple()

        # move from cube to hover
        T_e5_e6 = SE3.toSE3(SO3.I(), np.array([[0], [0], [-self.standoff]]))
        T_s_e6 = SE3.multiplication(T_s_e5, T_e5_e6)

        self.TrajMoveTo(T_s_e5, T_s_e6, 0)

        self.csv.WriteTrajectoryToCSV()

    def PIControl(self, kp, ki, step):
        
        KP = np.diag(np.array([1, 1, 1, 1, 1, 1]))*kp
        KI = np.diag(np.array([1, 1, 1, 1, 1, 1]))*ki

        X_e_hat = mr.MatrixLog6(SE3.multiplication(mr.TransInv(self.youBot.T_s_e), self.trajectory[step]))  # se3 representation
        X_e = np.array([mr.se3ToVec(X_e_hat)])      # twist representation of the error

        V_d_hat = mr.MatrixLog6(SE3.multiplication(mr.TransInv(self.trajectory[step]), self.trajectory[step+1]))*(1/self.youBot.dt)
        V_d = np.array([mr.se3ToVec(V_d_hat)])      # twist representation of the desired twist

        V_d_ff = np.matmul(mr.Adjoint(SE3.multiplication(mr.TransInv(self.youBot.T_s_e), self.trajectory[step])), V_d.transpose())        # feedforward twist Vd in the actual end effector frame

        integral = np.add(self.integral_prior, X_e.transpose()*self.youBot.dt)
        
        V_b = np.add(np.add(V_d_ff, np.matmul(KP, X_e.transpose())), np.matmul(KI, integral))       # should be a 6-vector

        self.integral_prior = integral
        self.error_prior = X_e.transpose()

        self.errorCsv.AddState(X_e.flatten())

        return V_b

    def testTraj(self):
        self.trajectory += [np.array([[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]]), np.array([[0, 0, 1, 0.6], [0, 1, 0, 0], [-1, 0, 0, 0.3], [0, 0, 0, 1]])]

    def Simulate(self, T_s_c_i, T_s_c_f, velocityControlLimits):
        
        # generate trajectory
        self.GenerateTrajectory_MoveBlock(T_s_c_i, T_s_c_f)
        #self.testTraj()
        print(self.youBot.T_s_e)
        # change initial config of youBot
        self.youBot = youBot.youBot(wheelPositions=None, chassisConfig=np.array([[0], [0], [0.2]]), planarTwist=None, jointConfig=np.array([[0],[-0.1],[0],[0.6],[0]]))
        print(self.youBot.T_s_e)

        for step in range(len(self.trajectory)-1):
            V_b = self.PIControl(10, 30, step)

            velocityControlInput = np.matmul(np.linalg.pinv(self.youBot.J_b), V_b)
            self.youBot.Step(velCtrlInput=velocityControlInput, velCtrlLimits=velocityControlLimits, gripperInput=self.grapplerTraj[step], isWritten=True)

        self.youBot.csv.WriteStateToCSV()
        self.errorCsv.plotState()

