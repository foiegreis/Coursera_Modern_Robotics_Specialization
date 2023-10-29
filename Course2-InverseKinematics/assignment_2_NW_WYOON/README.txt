About the Project:

In this project, the primary focus is on the smooth movement of the robot during the execution of inverse kinematics. 
Rapid and abrupt movements, especially from incorrect initial conditions, can be detrimental. 
Therefore, a dynamic learning rate is introduced, modeled using a Mexican Hat function, 
to ensure that the adjustments made in each iteration are smooth and effective.

Key Observations:
1. The dynamic learning rate using the Mexican Hat function has proven effective in providing smoother robot movements.

2. During experimentation, it was observed that the robot can encounter singularities, 
especially when the robot's arm is fully stretched in a linear configuration. 
Care should be taken during such configurations, as the usual inverse kinematics methods might not produce accurate results.
