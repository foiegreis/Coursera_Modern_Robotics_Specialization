import numpy as np
from functools import reduce
from utils import adjoint_of_SE3, screw_theta_to_SE3

class Frame:
    def __init__(self, position=np.zeros(3), orientation=np.eye(3)):
        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def update_position(self, position):
        self.position = np.array(position)

    def update_orientation(self, orientation):
        self.orientation = np.array(orientation)


class Robot:
    def __init__(self):
        self.joint_names = []
        self.joints = {}
        self.base_frame = Frame()
        self.end_effector_frame = Frame()

    def register_joint(self, joint):
        self.joints[joint.name] = joint
        self.joint_names.append(joint.name)

    def update_joint_thetas(self, joint_thetas):
        for joint_name, theta in joint_thetas.items():
            if joint_name not in self.joints:
                raise ValueError(f"No joint named {joint_name} in the robot.")
            
            self.joints[joint_name].update_theta(theta)

    def get_screw_axis(self, joint_name, isSpace=True):
        if joint_name not in self.joints:
            raise ValueError(f"No joint named {joint_name} in the robot.")
        
        reference_frame = self.base_frame if isSpace else self.end_effector_frame

        return self.joints[joint_name].compute_screw_axis(reference_frame)

    def get_joint_SE3(self, joint_name, isSpace=True, isInverse=False):
        if joint_name not in self.joints:
            raise ValueError(f"No joint named {joint_name} in the robot.")
        
        theta = self.joints[joint_name].theta
        screw_axis = self.get_screw_axis(joint_name, isSpace)
        
        return screw_theta_to_SE3(screw_axis, theta, isInverse)

    def get_matrix_M(self):
        E, B = self.end_effector_frame.orientation, self.base_frame.orientation
        p_target, p_source = self.end_effector_frame.position, self.base_frame.position
    
        R = np.dot(B, E.T)
        p = np.subtract(p_target, p_source)
        
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p
        return M
        
    def get_total_SE3(self, joint_thetas, isSpace=True):
        self.update_joint_thetas(joint_thetas)
        
        SE3_matrices = [self.get_joint_SE3(joint_name, isSpace) for joint_name in self.joint_names]
        SE3_matrices.append(self.get_matrix_M()) if isSpace else SE3_matrices.insert(0, self.get_matrix_M())
        total_SE3 = reduce(np.matmul, SE3_matrices)

        return total_SE3

    def get_jacobian_base(self, joint_thetas, isSpace=True):
        self.update_joint_thetas(joint_thetas)

        screw_axes = [self.get_screw_axis(joint_name, isSpace) for joint_name in self.joint_names]

        isInverse = (not isSpace)
        SE3_matrices = [self.get_joint_SE3(joint_name, isSpace, isInverse=isInverse) for joint_name in self.joint_names]
        
        traverse_order = range(len(SE3_matrices)) if isSpace else range(len(SE3_matrices))[::-1]

        jacobian_columns = []
        cur_SE3 = np.eye(4)
        for i in traverse_order:
            adjoint_SE3 = adjoint_of_SE3(cur_SE3)
            jacobian_col = np.dot(adjoint_SE3, screw_axes[i]).reshape(-1,1)
            jacobian_columns.append(jacobian_col)

            cur_SE3 = np.dot(cur_SE3, SE3_matrices[i])

        if not isSpace:
            jacobian_columns = (jacobian_columns)[::-1]
            
        J_B = np.hstack(jacobian_columns)
        return J_B