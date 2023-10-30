----------------------------------------------------------------------------------------------------
Code for the Kuka YouBot Pick and Place task, from Modern Robotics Course 6 on Coursera
BEST CASE

Config values:

cube initial config           phi = 0, x = 1, y = 0
cube final config             phi = -1.5707963267948966, x = 0, y = -1
initial config error          phi = -0.5235987755982988, x = -0.2, y = 0.2
initial config                q = [-2.58070105e-01 -1.62092975e-01 -4.59884828e-01 -1.31521716e+00
                                   -1.25768676e+00 -6.21912361e-03 -8.31461317e-01  1.44373441e-03
                                    2.08640605e+01 -1.20075576e+01  1.26247036e+01 -1.87299368e+00]
max wheels velocity           30
max arm joints velocity       30
controller type               ff_pi, kp = 8.0, kd = 0, ki = 0.9
test joint limits             True
steps in trajectory           1800
joint limits                  [[-0.52359878  0.52359878]
                               [-3.          0.        ]
                               [-4.          0.        ]
                               [-3.          0.        ]
                               [-1.57079633  0.        ]]


You will find the full configuration in code/config_best.ini

