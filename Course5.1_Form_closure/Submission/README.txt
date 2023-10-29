Thank you for evaluating this submission.  
I have written my solution in Python 3.10, using the suggested scipy approach for linear optimization

This solution uses the following input files
    polygon.csv: a list of the polygon points in csv format
    contacts_open.csv:  the set of contact points and normal angles for an open case (ie not in form closure)
    contacts_closed.csv: the set of contact points and normal angles for a closed case (is in closed form)

The solution uses matplotlib to generate the required images for each use case. these are exported as the following files

  results\contacts_open.png  an image of the open (non closed) use case
  results\contacts_closed.png an image of the closed (form closure) use case

The program runs both uses cases for and therefore does not return a value. However, the function
   calculate_form_closure(wrenches)
 does return True or False if the body is in (or not in) form closure.

Please note that I have used different contact points in my test scenario from the points given on the wiki.

An example run of the program is given below
--------------------------------------------------------------------------------------------
PS F:\Study\ModernRoboticsNW\Course5\Week1\FormClosure\code> python FormClosure.py
Evaluating form closure for ../contacts_open.csv

Wrench Matrix:
[[ 2.50000000e+01  3.50000000e+01 -4.00000000e+01 -1.41421356e+01]
 [ 6.12323400e-17  6.12323400e-17  1.00000000e+00 -7.07106781e-01]
 [ 1.00000000e+00  1.00000000e+00  0.00000000e+00 -7.07106781e-01]]
At least one open COR region found!
Rank of wrench matrix is 3
 Full closure is FALSE

Evaluating form closure for ../contacts_closed.csv

Wrench Matrix:
[[ 2.50000000e+01  3.50000000e+01 -4.00000000e+01 -3.26300775e+00]
 [ 6.12323400e-17  6.12323400e-17  1.00000000e+00 -5.73576436e-01]
 [ 1.00000000e+00  1.00000000e+00  0.00000000e+00 -8.19152044e-01]]
Rank of wrench matrix is 3
 Full closure is TRUE

PS F:\Study\ModernRoboticsNW\Course5\Week1\FormClosure\code> 