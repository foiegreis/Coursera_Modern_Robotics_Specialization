'''
Capstone project "Kuka YouBot mobile manipulator - pick and place task
Copyright @foiegreis 2023
'''

import matplotlib.pyplot as plt
import modern_robotics as mr
import numpy as np
import pandas as pd
import configparser
import argparse
import sys
import os

class YouBot:
    def __init__(self, d_l, d_w, r, m_0e, b, t_b0, joint_limits):
        self.d_l = d_l  # half length of mobile base
        self.d_w = d_w  # half wheels distance
        self.r = r  # wheel radius

        self.m_0e = m_0e
        self.b = b
        self.t_b0 = t_b0

        self.f = np.dot((self.r / 4), np.array([[-1 / (self.d_l + self.d_w), 1 / (self.d_l + self.d_w),
                                                 1 / (self.d_l + self.d_w), -1 / (self.d_l + self.d_w)],
                                                [1, 1, 1, 1],
                                                [-1, 1, -1, 1]]))

        self.f6 = np.zeros([6, self.f.shape[1]])
        self.f6[2:5, :] = self.f[0:3, :]

        self.joint_limits = joint_limits

class PickAndPlace(YouBot):
    def __init__(self, dt, k, method, degree, z_standoff, pickup_angle, max_w_dot, max_a_dot,
                 control_type, test_joint_limits_flag):
        super().__init__(d_l, d_w, r, m_0e, b, t_b0, joint_limits)

        self.dt = dt  # timestep
        self.k = k  # how many trajectory points for each timestep

        self.method = method  # 'skew' or 'cartesian
        self.degree = degree  # 3 for cubic, 5 for quintic

        self.z_standoff = z_standoff  # z-axis standoff
        self.pickup_angle = pickup_angle

        self.max_w_dot = max_w_dot  # max wheel velocity
        self.max_a_dot = max_a_dot  # max arm joint velocity

        self.control_type = control_type  # 'ff_pi' or 'ff_pid'

        self.test_joint_limits_flag = test_joint_limits_flag

    @staticmethod
    def decode_configuration(q):
        """Decodes configuration q to: qc (chassis config), qa (arms config), qw (wheels config)"""
        return np.array(q[0:3]), np.array(q[3:8]), np.array(q[8:12])

    @staticmethod
    def decode_control(u):
        """Decodes control u to: qw_dot, qa_dot"""
        return np.array(u[0:4]), np.array(u[4:9])

    def vel_limits(self, qw_dot, qa_dot):
        """Checks boundaries of max and min velocities for wheels and arm joints"""
        qw_dot[qw_dot > self.max_w_dot] = self.max_w_dot
        qw_dot[qw_dot < - self.max_w_dot] = - self.max_w_dot

        qa_dot[qa_dot > self.max_a_dot] = self.max_a_dot
        qa_dot[qa_dot < - self.max_a_dot] = - self.max_a_dot
        return qw_dot, qa_dot

    def odometry(self, qc, qw_delta):
        """Computes odometry to get new chassis configuration from new wheel movement"""
        phi, x, y = qc

        twist = np.dot(self.f, qw_delta)
        wbz, vbx, vby = twist

        if wbz < 1e-9:  # wbz is considered zero if < 1e-9
            qc_delta = [0, vbx, vby]
        else:
            qphi_delta = wbz
            qx_delta = (vbx * np.sin(wbz) + vby * (np.cos(wbz) - 1)) / wbz
            qy_delta = (vby * np.sin(wbz) + vbx * (1 - np.cos(wbz))) / wbz
            qc_delta = [qphi_delta, qx_delta, qy_delta]

        t_sb = np.array([[1, 0, 0],
                         [0, np.cos(phi), -np.sin(phi)],
                         [0, np.sin(phi), np.cos(phi)]])

        q_delta = np.dot(t_sb, qc_delta)

        qc_new = qc + q_delta
        return qc_new

    def next_state(self, q, u):
        """Projects configuration q to next timestep
        :param q = qc(chassis phi, chassis x, chassis y), qa(J1, J2, J3, J4, J5), qw(W1, W2, W3, W4)
        :param u = qw_dot, qa_dot
        :return qc_new = new chassis config
        """

        # get configurations
        qc, qa, qw = self.decode_configuration(q)
        qw_dot, qa_dot = self.decode_control(u)

        # check velocities limits
        qw_dot, qa_dot = self.vel_limits(qw_dot, qa_dot)

        # euler step
        qa_delta = qa_dot * self.dt
        qw_delta = qw_dot * self.dt

        qa_new = qa + qa_delta
        qw_new = qw + qw_delta

        # odometry
        qc_new = self.odometry(qc, qw_delta)

        q_new = np.concatenate([qc_new, qa_new, qw_new])
        return q_new

    @staticmethod
    def encode_trajectory(traj, gripper_state):
        """Parses trajectory as: [r11 r12 r13 r21 r22 r23 r31 r32 r33 px py pz gripper_state]"""
        traj_out = []
        for T in traj:
            t = []
            R, p = mr.TransToRp(T)
            for i, row in enumerate(R):
                for el in row:
                    t.append(el)
            for j in p:
                t.append(j)
            t.append(gripper_state)
            traj_out.append(t)
        return traj_out

    @staticmethod
    def rotate_matrix(R, phi, axis):
        """Rotates R by angle phi about desired axis"""
        rot = None
        if phi == 0:
            rot = np.eye(3)
        else:
            if axis == 'x':
                rot = [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
            if axis == 'y':
                rot = [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
            elif axis == 'z':
                rot = [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
        R_new = np.dot(rot, R)
        return R_new

    def trajectory_segment(self, x_start, x_end, tf, gripper_state):
        """Computes trajectory segment with mr.ScrewTrajectory or mr.CartesianTrajectory method
        :param x_start = Tse_init
        :param x_end = Tse_final
        :param tf = timestep
        :param gripper_state
        :return trajectory segment as vector of 13 elements
        """

        trajectory = None
        if x_end is None:  # in case of Grab or Release
            trajectory = [x_start for _ in range(0, int(tf * k / 0.01))]
        else:
            N = (tf * self.k) / 0.01
            if self.method == 'screw':
                trajectory = mr.ScrewTrajectory(x_start, x_end, tf, N, self.degree)
            elif self.method == 'cartesian':
                trajectory = mr.CartesianTrajectory(x_start, x_end, tf, N, self.degree)

        traj_13 = self.encode_trajectory(trajectory, gripper_state)
        return traj_13

    def segment_duration(self, x_start, x_end):
        """Computes the optimal duration in seconds for a trajectory segment x_start to x_end. Min duration 1s"""

        R_start, p_start = mr.TransToRp(x_start)
        R_end, p_end = mr.TransToRp(x_end)

        phi_start, x_start, y_start = p_start
        phi_end, x_end, y_end = p_end

        v = 1 / 4  # 0.25 m/s
        w = 1 / 4  # 0.25 rad/s

        d = np.sqrt((phi_end - phi_start) ** 2 + (x_end - x_start) ** 2 + (y_end - y_start) ** 2)
        theta = np.arccos((np.trace(R_end) - 1) / 2) - np.arccos((np.trace(R_start) - 1) / 2)

        linear_t = d / v
        angular_t = theta / w

        res = max(angular_t, linear_t)
        return 1 if res < 1 else int(res)

    def trajectory_generator(self, tse_init, tsc_init, tsc_final, shift_x, shift_z):
        """Reference Trajectory Generation via point to point
        :param tse_init = initial end effector configuration
        :param tsc_init = initial cube configuration
        :param tsc_final = final cube configuration
        :param shift_x = shift in x{s} direction for better grasping
        :param shift_z = shift in z{s} direction for better grasping
        :return desired_trajectory
        """

        Rse_init, pse_init = mr.TransToRp(tse_init)
        Rc_init, pc_init = mr.TransToRp(tsc_init)
        Rc_final, pc_final = mr.TransToRp(tsc_final)

        R1 = self.rotate_matrix(Rse_init, self.pickup_angle, 'y')
        p1 = [pc_init[0], pc_init[1], pc_init[2] + self.z_standoff]

        R2 = R1
        # shift to grab the cube more firmely
        p2 = [pc_init[0] + shift_x, pc_init[1], pc_init[2] + shift_z]

        R5 = self.rotate_matrix(R2, -np.pi / 2, 'z')
        p5 = [pc_final[0], pc_final[1], pc_final[2] + self.z_standoff]

        R6 = R5
        # here we subtract shift_x and shift_z to re-set the correct goal position
        p6 = [pc_final[0], pc_final[1] - shift_x, pc_final[2] - shift_z]

        tse_init_standoff = mr.RpToTrans(R1, p1)
        tse_init_grab = mr.RpToTrans(R2, p2)
        tse_final_standoff = mr.RpToTrans(R5, p5)
        tse_final_release = mr.RpToTrans(R6, p6)

        gripper_state = 0

        # 1: tse_init - tse_init_standoff
        # tf = 4
        tf = self.segment_duration(tse_init, tse_init_standoff)
        traj1 = self.trajectory_segment(tse_init, tse_init_standoff, tf, gripper_state)

        # 2: tse_init_standoff - tse_init_grab
        # tf = 1
        tf = self.segment_duration(tse_init_standoff, tse_init_grab)
        traj2 = self.trajectory_segment(tse_init_standoff, tse_init_grab, tf, gripper_state)

        gripper_state = 1

        # 3: grab
        tf = 2  # fixed time for grab
        traj3 = self.trajectory_segment(tse_init_grab, None, tf, gripper_state)

        # 4: tse_init_grab - tse_init_standoff
        # tf = 1
        tf = self.segment_duration(tse_init_grab, tse_init_standoff)
        traj4 = self.trajectory_segment(tse_init_grab, tse_init_standoff, tf, gripper_state)

        # 5: tse_init_standoff - tse_final_standoff
        # tf = 5
        tf = self.segment_duration(tse_init_standoff, tse_final_standoff)
        traj5 = self.trajectory_segment(tse_init_standoff, tse_final_standoff, tf, gripper_state)

        # 6: tse_final_standoff - tse_final_release
        # tf = 3
        tf = self.segment_duration(tse_final_standoff, tse_final_release) + 1
        traj6 = self.trajectory_segment(tse_final_standoff, tse_final_release, tf, gripper_state)

        gripper_state = 0

        # 7: release
        tf = 2  # fixed time for release
        traj7 = self.trajectory_segment(tse_final_release, None, tf, gripper_state)

        # 8: tse_final_release - tse_final_standoff
        # tf = 1
        tf = self.segment_duration(tse_final_release, tse_final_standoff)
        traj8 = self.trajectory_segment(tse_final_release, tse_final_standoff, tf, gripper_state)

        # concatenate segments
        desired_trajectory = np.concatenate((traj1, traj2, traj3, traj4, traj5, traj6, traj7, traj8))

        return desired_trajectory

    @staticmethod
    def decode_trajectory(traj):
        """Decodes trajectory from vector of 13 elements to X matrix, gripper_state"""
        r, p, gripper_state = traj[0: 9], traj[9:12], traj[12]
        R = list(zip(*(iter(r),) * 3))
        X = mr.RpToTrans(R, p)
        return X, gripper_state

    def forward_kinematics_se(self, qc, qa):
        """Computes Tse matrix
        :param qc = chassis configuration
        :param qa = arm joints configuration
        :return t_se
        """
        phi, x, y = qc
        t_sb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                         [np.sin(phi), np.cos(phi), 0, y],
                         [0, 0, 1, 0.0963],
                         [0, 0, 0, 1]])

        t_0e = mr.FKinBody(self.m_0e, self.b, qa)
        t_s0 = np.dot(t_sb, self.t_b0)
        t_se = np.dot(t_s0, t_0e)
        return t_se

    def inverse_kinematics(self, t_0e, qa):
        """Computes inverse kinematics
        :param t_0e = matrix TOe
        :param qa = arm joints configuration
        :return theta_list = list of joint angles
        """
        eps_w = 0.001
        eps_v = 0.0001
        [theta_list, success] = mr.IKinBody(self.b, self.m_0e, t_0e, qa, eps_w, eps_v)
        if not success:
            raise ValueError('Inverse kinematics failed')
        return theta_list

    def feedforward_pid_control(self, t_se, t_se_d, t_se_d_next, kp, kd, ki, timestep, xe_integral, xe_prev):
        """Computes FeedForward + PID control
        :param t_se
        :param t_se_d = desired Tse at step k
        :param t_se_d_next = desired Tse at step k+1
        :param kp = proportional gain
        :param kd = derivative gain
        :param ki = integral gain
        :param timestep
        :return v = control twist
        :return x_err = current error
        """
        # desired twist
        vd = mr.se3ToVec(np.dot((1 / timestep), mr.MatrixLog6(np.dot(mr.TransInv(t_se_d), t_se_d_next))))

        # config error
        x_err = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(t_se), t_se_d)))

        # feedforward control
        feedforward = np.dot(mr.Adjoint(np.dot(mr.TransInv(t_se), t_se_d)), vd)

        # error derivative
        xe_derivative = (x_err - xe_prev) / timestep

        # control twist
        v = feedforward + np.dot(kp, x_err) + np.dot(kd, xe_derivative) + np.dot(ki, (xe_integral * timestep))

        xe_integral += x_err
        xe_prev = x_err
        return v, x_err, xe_integral, xe_prev

    def feedforward_pi_control(self, t_se, t_se_d, t_se_d_next, kp, ki, timestep, xe_integral):
        """Computes FeedForward + PI control
        :param t_se
        :param t_se_d =  desired Tse at step k
        :param t_se_d_next =  desired Tse at step k+1
        :param kp = proportional gain
        :param ki = integral gain
        :param timestep
        :return v =  control twist
        :return x_err =  current error
        """
        # desired twist
        vd = mr.se3ToVec(np.dot((1 / timestep), mr.MatrixLog6(np.dot(mr.TransInv(t_se_d), t_se_d_next))))

        # config error
        x_err = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(t_se), t_se_d)))

        # feedforward control
        feedforward = np.dot(mr.Adjoint(np.dot(mr.TransInv(t_se), t_se_d)), vd)

        # control twist
        v = feedforward + np.dot(kp, x_err) + np.dot(ki, (xe_integral * timestep))

        xe_integral += x_err
        return v, x_err, xe_integral

    def pi_control(self, t_se, t_se_d, kp, ki, timestep, xe_integral):
        """Computes PI control
        :param t_se
        :param t_se_d =  desired Tse at step k
        :param t_se_d_next =  desired Tse at step k+1
        :param kp = proportional gain
        :param ki = integral gain
        :param timestep
        :return v =  control twist
        :return x_err =  current error
        """
        # config error
        x_err = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(t_se), t_se_d)))

        # control twist
        v = np.dot(kp, x_err) + np.dot(ki, (xe_integral * timestep))

        xe_integral += x_err
        return v, x_err, xe_integral

    def compute_je(self, qa):
        """Computes end effector Jacobian"""

        t_0e = mr.FKinBody(self.m_0e, self.b, qa)
        j_base = np.dot(mr.Adjoint(np.dot(mr.TransInv(t_0e), mr.TransInv(self.t_b0))), self.f6)
        j_arm = mr.JacobianBody(self.b, qa)
        j_e = np.concatenate((j_base, j_arm), axis=1)
        return j_e

    def test_joint_limits(self, q, j):
        """Checks if J respects the constraints
        you could constrain joints 3 and 4 to always be less than -0.2 radians (or so).
        The arm will avoid singularities occurring when joints 3 or 4 are at the zero angle,
        but it will still be able to perform many useful tasks
        To recalculate the controls, change each column of Je corresponding to an offending joint to all zeros.
        This indicates that moving these joints causes no motion at the end-effector, so the pseudoinverse
        solution will not request any motion from these joints.
        """

        q_check = q[3:8]  # arm joints
        j_check = j.copy()

        for i, qi in enumerate(q_check):
            if self.joint_limits[i][0] != 0 and qi <= self.joint_limits[i][0]:
                j_check[:, i] = 0
            if self.joint_limits[i][1] != 0 and qi >= self.joint_limits[i][1]:
                j_check[:, i] = 0
        return j_check

    def compute_velocities_control(self, q, qc, qa, xd, xd_next, kp, kd, ki, dt, xe_integral, xe_prev):
        """Computes U, the velocity control
        :param q = current configuration (12 elements)
        :param qc = current chassis configuration
        :param qa = current arm configuration
        :param xd = current desired configuration
        :param xd_next = next desired configuration
        :param kp = proportional gain
        :param kd = derivative gain
        :param ki = integral gain
        :param timestep
        :return u = velocity control
        :return x_err = conf error
        """
        v_e, x_err = None, None
        x = self.forward_kinematics_se(qc, qa)  # x = t_se

        if self.control_type == 'ff_pi':
            v_e, x_err, xe_integral = self.feedforward_pi_control(x, xd, xd_next, kp, ki, dt, xe_integral)
        elif self.control_type == 'ff_pid':
            v_e, x_err, xe_integral, xe_prev = self.feedforward_pid_control(x, xd, xd_next, kp, kd, ki, dt, xe_integral, xe_prev)
        elif self.control_type == 'pi':
            v_e, x_err, xe_integral = self.pi_control(x, xd, kp, ki, dt, xe_integral)

        # compute end-effector jacobian
        j_e = self.compute_je(qa)
        if self.test_joint_limits_flag:
            j_e = self.test_joint_limits(q, j_e)

        # pseudo inverse tolerance
        tol = 0.0001

        # velocity control
        u = np.dot(np.linalg.pinv(j_e, tol), v_e)

        return u, x_err, xe_integral, xe_prev

    def plot_and_save_error(self, path, name):
        """Plots error"""
        print("Plotting error.")
        with open(path, 'r') as csvfile:
            df = pd.read_csv(csvfile, delimiter=',')
            err1 = df.iloc[:, 0]
            err2 = df.iloc[:, 1]
            err3 = df.iloc[:, 2]
            err4 = df.iloc[:, 3]
            err5 = df.iloc[:, 4]
            step = list(range(df.shape[0]))

        plt.plot(step, err1, label='j1')
        plt.plot(step, err2, label='j2')
        plt.plot(step, err3, label='j3')
        plt.plot(step, err4, label='j4')
        plt.plot(step, err5, label='j5')
        plt.legend(loc='upper right')
        plt.xlabel('Time (ms)')
        plt.ylabel(f'Error - se(3) for controller {self.control_type}')
        plt.savefig(name)
        plt.show()

    def plot_and_save_controls(self, path, name):
        """Plots controls"""
        with open(path, 'r') as csvfile:
            df = pd.read_csv(csvfile, delimiter=',')
            u1 = df.iloc[:, 0]
            u2 = df.iloc[:, 1]
            u3 = df.iloc[:, 2]
            u4 = df.iloc[:, 3]
            u5 = df.iloc[:, 4]
            u6 = df.iloc[:, 5]
            u7 = df.iloc[:, 6]
            u8 = df.iloc[:, 7]
            u9 = df.iloc[:, 8]
            step = list(range(df.shape[0]))

        plt.plot(step, u1, step, u2, step, u3, step, u4, step, u5, step, u6, step, u7, step, u8, step, u9)
        plt.xlabel('Time (ms)')
        plt.ylabel(f'Control - {self.control_type}')
        plt.savefig(name)
        plt.show()


###################################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", required=True, help='execution mode: best, overshoot, newTask')
    args = parser.parse_args()
    suffix = '_' + args.mode

    config = configparser.ConfigParser()
    config.read(f'config{suffix}.ini')

    print(f"\n**** Capstone project - YouBot pick And place task  - Testcase: {suffix[1:]} ****")

    # ---------------------------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------------------------
    d_l = eval(config['youbot']['d_l'])
    d_w = eval(config['youbot']['d_w'])
    r = eval(config['youbot']['r'])

    m_0e = eval(config['youbot']['m_0e'])
    b = eval(config['youbot']['b'])
    t_b0 = eval(config['youbot']['t_b0'])

    # ---------------------------------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------------------------------
    joint_limits = eval(config['youbot']['joint_limits'])

    max_w_dot = eval(config['control']['max_w_dot'])  # max wheels velocity
    max_a_dot = eval(config['control']['max_a_dot'])  # max arm joints velocity

    dt = eval(config['control']['dt'])
    k = eval(config['control']['k'])

    method = eval(config['control']['method'])
    degree = eval(config['control']['degree'])

    z_standoff = eval(config['control']['z_standoff'])
    pickup_angle = eval(config['control']['pickup_angle'])

    # we add shift x and shift z to tsc_init and final to grab the cube more firmly
    # p_grab = [pc_init[0] + shift_x, pc_init[1], pc_init[2] + shift_z]
    # p_release = [pc_final[0], pc_final[1] - shift_x, pc_final[2] - shift_z]
    shift_x = eval(config['control']['shift_x'])
    shift_z = eval(config['control']['shift_z'])

    kp = eval(config['control']['kp'])
    kd = eval(config['control']['kd'])
    ki = eval(config['control']['ki'])

    control_types = eval(config['control']['control_types'])
    control_type = eval(config['control']['control_type'])

    test_joint_limits_flag = eval(config['control']['test_joint_limits_flag'])

    # initial trajectory configuration
    tse_init = eval(config['control']['tse_init'])

    # cube configurations
    phi_init = eval(config['control']['phi_init'])
    x_init = eval(config['control']['x_init'])
    y_init = eval(config['control']['y_init'])
    tsc_init = eval(config['control']['tsc_init'])

    phi_final = eval(config['control']['phi_final'])
    x_final = eval(config['control']['x_final'])
    y_final = eval(config['control']['y_final'])
    tsc_final = eval(config['control']['tsc_final'])

    # initial end effector configuration error from tse_init
    err_phi = eval(config['control']['err_phi'])
    err_x = eval(config['control']['err_x'])
    err_y = eval(config['control']['err_y'])
    theta1 = eval(config['control']['theta1'])
    theta2 = eval(config['control']['theta2'])
    theta3 = eval(config['control']['theta3'])

    # chassis_phi, chassis_x, chassis_y, j1, j2, j3, j4, j5, w1, w2, w3, w4
    q = eval(config['control']['q'])

    # initial control
    u = eval(config['control']['u'])

    # initial gripper state
    gripper_state = eval(config['control']['gripper_state'])

    # ---------------------------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------------------------
    x_list = []
    x_err_list = []
    u_list = []

    q_save = q.copy()
    q_save.append(gripper_state)
    x_list.append(np.array(q_save))
    u_list.append(u)

    # initialize
    robot = YouBot(d_l, d_w, r, m_0e, b, t_b0, joint_limits)
    controller = PickAndPlace(dt, k, method, degree, z_standoff, pickup_angle, max_w_dot, max_a_dot,
                              control_type, test_joint_limits_flag)

    # generate desired trajectory
    planned_trajectory = controller.trajectory_generator(tse_init, tsc_init, tsc_final, shift_x, shift_z)

    xe_integral = np.zeros(6)  # to calculate integration
    xe_prev = np.zeros(6)  # to calculate derivative

    for i in range(len(planned_trajectory)-1):

        # get desired configurations
        xd, xd_gripper_state = controller.decode_trajectory(planned_trajectory[i])
        xd_next, xd_next_gripper_state = controller.decode_trajectory(planned_trajectory[i+1])

        qc, qa, qw = controller.decode_configuration(q)

        # compute control
        u, x_err, xe_integral, xe_prev = controller.compute_velocities_control(q, qc, qa, xd, xd_next,
                                                                               kp, kd, ki, dt,
                                                                               xe_integral, xe_prev)

        # generate next state
        q = controller.next_state(q, u)

        q_save = q.copy()
        q_save = np.concatenate([q_save, np.array([xd_gripper_state])])
        x_list.append(q_save)
        x_err_list.append(x_err)
        u_list.append(u)

    path_traj_csv = os.path.join(os.path.abspath('..'), f'results/{suffix[1:]}/trajectory{suffix}.csv')
    path_err_csv = os.path.join(os.path.abspath('..'), f'results/{suffix[1:]}/err{suffix}.csv')
    path_u_csv = os.path.join(os.path.abspath('..'), f'results/{suffix[1:]}/controls{suffix}.csv')
    path_td_csv = os.path.join(os.path.abspath('..'), f'results/{suffix[1:]}/desired_trajectory{suffix}.csv')

    df = pd.DataFrame(x_list)
    df.to_csv(path_traj_csv, header=False, index=False)

    df = pd.DataFrame(x_err_list)
    df.to_csv(path_err_csv, header=False, index=False)

    df = pd.DataFrame(u_list)
    df.to_csv(path_u_csv, header=False, index=False)

    df = pd.DataFrame(planned_trajectory)
    df.to_csv(path_td_csv, header=False, index=False)

    # ---------------------------------------------------------------------------------------------
    # Logging to log.txt file
    # ---------------------------------------------------------------------------------------------
    temp = sys.stdout
    sys.stdout = open(os.path.join(os.path.abspath('..'), f'results/{suffix[1:]}/log{suffix}.txt'), 'w')

    print(f"Config{suffix}")
    print(f"cube initial config           phi = {phi_init}, x = {x_init}, y = {y_init}")
    print(f"cube final config             phi = {phi_final}, x = {x_final}, y = {y_final}")
    print(f"initial config error          phi = {err_phi}, x = {err_x}, y = {err_y}")
    print(f"initial config                q = {q}")
    print(f"max wheels velocity           {max_w_dot}")
    print(f"max arm joints velocity       {max_a_dot}")
    print(f"controller type               {control_type}, kp = {kp[0][0]}, kd = {kd[0][0] if control_type=='ff_pid' else 0}, ki = {ki[0][0]}")
    print(f"test joint limits             {test_joint_limits_flag}")
    print(f"steps in trajectory           {len(planned_trajectory)}")
    print(f"joint limits\n{joint_limits}")
    sys.stdout.close()
    sys.stdout = temp

    controller.plot_and_save_error(path_err_csv, path_err_csv.split('.')[0]+'.png')
    controller.plot_and_save_controls(path_u_csv, path_u_csv.split('.')[0]+'.png')

    print(f"Coppeliasim path         {path_traj_csv} ")
    print("Run configurations saved to log.txt file.")
    print("Done.")




