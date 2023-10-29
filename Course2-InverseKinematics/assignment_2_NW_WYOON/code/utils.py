import numpy as np

def vector_to_so3(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def omega_theta_to_SO3(omega, theta):
    omega_cross = vector_to_so3(omega)
    return np.eye(3) + np.sin(theta) * omega_cross + (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)

def omega_v_theta_to_translation(omega, theta, v):
    omega_cross = vector_to_so3(omega)
    p = np.eye(3)*theta + (1 - np.cos(theta)) * omega_cross + (theta - np.sin(theta))* np.dot(omega_cross, omega_cross)
    return np.matmul(p,v)

def screw_theta_to_SE3(screw, theta, isInverse=False):
    if isInverse:
        screw = -screw
    
    omega = screw[:3]
    v = screw[3:]
    R = omega_theta_to_SO3(omega, theta)
    p = omega_v_theta_to_translation(omega, theta, v)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def adjoint_of_SE3(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_skew = vector_to_so3(p)

    top = np.hstack((R, np.zeros((3,3))))
    bottom = np.hstack((p_skew @ R, R))
    return np.vstack((top, bottom))

def extract_screw_from_logm(logm_result):
    w = np.array([
        logm_result[2, 1],
        logm_result[0, 2],
        logm_result[1, 0]
    ])

    v = logm_result[:3, 3]
    
    return w, v