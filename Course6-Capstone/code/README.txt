----------------------------------------------------------------------------------------------------
General code for the Kuka YouBot Pick and Place task, from Modern Robotics Course 6 on Coursera
copyrights @foiegreis,  October 2023

this is an example version, you'll find the results in the /results folder, and the code will run
on a different config.ini file for each case

The script generates:

log.txt -  with log stdout info

trajectory.csv - the generated trajectory
desired_trajectory.csv - the desired generated trajectory
err.csv - the error values for each timestep
controls.csv - the velocity control values for each timestep

err.png - the plot of the error values
control.png - the plot of the control values

Please refer to the code in the results /best /newTask /overshoot folders



