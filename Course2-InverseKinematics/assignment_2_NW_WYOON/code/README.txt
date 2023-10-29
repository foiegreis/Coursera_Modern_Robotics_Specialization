# Code Usage Guide

This work has evolved quite a bit from the original foundations. Initially, 
I took inspiration from the official lecture repository:
[ModernRobotics Python Core](https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py)
 
I modify and edit code during study of the lecture:
[My Robotics Study Chapter 4](https://github.com/littlepsilon/robotics_study/tree/main/chapter4)

Taking all that knowledge, here's how I've designed this assignment:

## 1. Configuration Setup:
Ensure that the settings for `T_sd` and `thetas_0` are adjusted in the "config.ini" file before anything else.

## 2. Running the Code:
Follow these steps for a seamless execution:
- Extract the configuration parameters: `T_sd, thetas_0 = parse_config("./config.ini")`
- Boot up the robot simulation: `rb = get_UR5_robot()`
- Set your preferred learning rate cycle and decide on the number of epochs you'd like to run.
- Invoke the Newton-Raphson inverse kinematics with: `newton_raphson_inverse_kinematics(...)`
- Finally, archive the results with `save_to_csv(thetas_history)`

## 3. Diving Deep into the Main Function:
The `newton_raphson_inverse_kinematics` function is the heart of this setup. It determines the joint angles via the Newton-Raphson technique. As it operates, each iteration's specifics get captured and logged. On completion, you get a compiled history of joint angles over the iterations.

### How it works?
1. You can run main.py using command python ./main.py
Or
2. You can run using jupyter notebook file main_notebook.ipynb
