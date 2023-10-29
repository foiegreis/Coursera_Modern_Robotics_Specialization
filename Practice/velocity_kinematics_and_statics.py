import modern_robotics as mr
import numpy as np
import scipy


#screw axes of RRP at zero position
Slist = np.array([[0,1,0],
                  [0,0,0],
                  [1,0,0],
                  [0,0,0],
                  [0,2,1],
                  [0,0,0]])

thetalist = np.array([np.pi/2, np.pi/2, 1])
Js = mr.JacobianSpace(Slist, thetalist)
print(np.round(Js, 3))


Blist = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,0],
                  [3,0,0],
                  [0,3,0],
                  [0,0,1]])
Jb = mr.JacobianBody(Blist, thetalist)
print(np.round(Jb, 3))



#Manipulability ellipsoid
Jbv = np.array([[-0.105, 0, 0.006, -0.045, 0, 0.006, 0],
[-0.889, 0.006, 0, -0.844, 0.006, 0, 0],
[0, -0.105, 0.889, 0, 0, 0, 0]])

invJbv = mr.RotInv(Jbv)

A = np.dot(Jbv, invJbv)

print("Manipulability ellipsoid ", np.round(A, 4), sep='\n\n', end='\n\n')
results = scipy.linalg.eig(A)
eig_values = results[0]
eig_vectors = results[1]

max_value = max(eig_values)
max_index = np.argmax(eig_values)
lon_direction = eig_vectors[:, max_index]
print('-----------')
print('lenght of longest principal semi-axis: ', np.real(np.round_(max_value, 3)))
print('direction of longest semi-axis: ', np.round_(lon_direction, 3))
