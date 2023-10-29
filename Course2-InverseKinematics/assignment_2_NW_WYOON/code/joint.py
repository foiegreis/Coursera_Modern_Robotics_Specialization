import numpy as np

class Joint:
    def __init__(self, name, position, orientation=np.eye(3)):
        self.name = name
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.theta = 0.0

    def update_theta(self, theta):
        self.theta = theta

    def compute_screw_axis(self):
        raise NotImplementedError("Method get_screw_axis should be implemented in subclasses.")


class RevoluteJoint(Joint):
    def __init__(self, name, position, rotation_axis=[0, 0, 1], orientation=np.eye(3)):
        """
        Joint allowing rotational motion around the specified rotation axis.
        """
        super().__init__(name, position, orientation)
        self.rotation_axis = np.array(rotation_axis)

    def compute_screw_axis(self, reference_frame=None):
        """
        Computes the screw axis for a revolute joint.
        If a reference frame is provided, the axis and position are transformed accordingly.
        """
        axis = self.rotation_axis
        joint_position = self.position
        
        if reference_frame:
            axis = np.dot(reference_frame.orientation, axis)
            joint_position = np.dot(reference_frame.orientation, joint_position - reference_frame.position)
        
        moment = -np.cross(axis, joint_position)
        return np.hstack([axis, moment])