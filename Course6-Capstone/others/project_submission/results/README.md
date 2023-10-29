# Project Illustration 
The controller type here is PI + feedforward controller, where the gain parameter is the same for each dimension.
This design methodology simplifies the overall tuning process and might not be the most efficient way. 
The detailed parameters are shown below: 

|Test Case|Cube Initial Position|Cube Goal Position|Proportional Gain|Integral Gain|
|-|-|-|-|-|
|overshoot|(1, 0, 0) | (0, -1, -pi/2) | 2.2 | 4 |
|best| (1, 0, 0) | (0, -1, -pi/2) | 4 | 0.5 |
|newTask|(0.8, 0.2, 0) | (0.5, -0.8, -pi/2) | 4 | 0.5|