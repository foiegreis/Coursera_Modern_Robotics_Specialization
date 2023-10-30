----------------------------------------------------------------------------------------------------
Code for the Kuka YouBot Pick and Place task, from Modern Robotics Course 6 on Coursera
NEW TASK CASE

Config values:

cube initial config           phi = 0, x = 0.5, y = -1
cube final config             phi = 1.5707963267948966, x = 1, y = 0
initial config error          phi = -0.2617993877991494, x = -0.1, y = 0.1
initial config                q = [ 6.11079667e-01  8.63134722e-01  4.04265364e-01 -2.18822678e+00
                                 -1.22517844e+00 -6.08794420e-02 -8.21267877e-01 -6.93205223e-04
                                  3.19727184e+01  1.93977122e+01  3.68880051e+01  3.95066557e+01]
max wheels velocity           30
max arm joints velocity       30
controller type               ff_pid, kp = 2.8, kd = 0.1, ki = 0.09
test joint limits             True
steps in trajectory           1700
joint limits                [[-0.3  0.3]
                             [-0.6  0. ]
                             [-0.8  0. ]
                             [ 0.   0. ]
                             [ 0.   0. ]]

You will find the full configuration in code/config_newTask.ini