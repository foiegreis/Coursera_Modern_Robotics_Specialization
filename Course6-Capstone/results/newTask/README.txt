----------------------------------------------------------------------------------------------------
Code for the Kuka YouBot Pick and Place task, from Modern Robotics Course 6 on Coursera
NEW TASK CASE

Config values:

cube initial config           phi = 0, x = 0, y = -1
cube final config             phi = 0, x = 0.5, y = 0.5
initial config error          phi = -0.2617993877991494, x = -0.1, y = 0.1
initial config                q = [-0.2617993877991494, -0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
max wheels velocity           20
max arm joints velocity       20
controller type               ff_pid, kp = 2.5, kd = 0.6, ki = 0.003
test joint limits             True
joint limits                  [-0.3, -0.2, -0.8, 0, 0]
steps in trajectory           1900

You will find the full configuration in config_newTask.ini