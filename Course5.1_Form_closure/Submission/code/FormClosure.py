#--------------------
# Modern Robotics Course 5 Week 1 Assigment
# First order form closure evaluation demo
# Keith Kitchin
#--------------------

import modern_robotics as mr
import numpy as np
import numpy.linalg as nplin
import csv
import math
import matplotlib.pyplot as plt
import scipy
import scipy.optimize


#--------------------- 
# Plot utilities
#-----------------

#plots the polygon points
def plot_polygon_points(polygon_point_matrix):
    x=polygon_point_matrix[:,0]
    y=polygon_point_matrix[:,1]
    plt.fill(x, y)

#plots the contact points 
def plot_contact_points(contact_point_matrix):
    x=contact_point_matrix[:,0]
    y=contact_point_matrix[:,1]
    plt.plot(x,y,'ro')

#plots the normal arrows
def plot_normals(contact_point_matrix, normals_matrix):
    x=contact_point_matrix[:,0]
    y=contact_point_matrix[:,1]
    dx=normals_matrix[:,0]*10
    dy=normals_matrix[:,1]*10
    for i in range (len(x)):

        plt.plot([x[i]-dx[i]*50,x[i]+dx[i]*50],[y[i]-dy[i]*50,y[i]+dy[i]*50], linewidth=0.5, linestyle=(0,(5,10)), color = "black")
        plt.arrow(x[i],y[i],dx[i],dy[i], head_width=2, head_length=3, fill=True, facecolor="black")
    
#determines if a point is 'left' or 'right' of a line
# from  https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
def get_point_side_of_line(x,y,px,py,nx,ny):
    AX=px
    BX=px+nx*10
    AY=py
    BY=py+ny*10
    sign = ((BX-AX)*(y-AY) - (BY-AY)*(x-AX) )
    if(sign > 0):
        return 1
    if(sign < 0):
         return -1
    return 0

#brute force plot our twist cone regions by testing each point in the domain
#against the normal lines. Use this to create a visual indication of any open COR regions
#NOTE - this is for visual debugging only and does not contribute to the form closure calculation
def plot_cor_regions(contact_point_matrix,normals_matrix):
    xstart = 0
    xend = 150
    xskip =1
    ystart =0
    yend = 150
    yskip =1
    num_points = len(contact_point_matrix)
    cor_polycone_found =False

    for x in range(xstart, xend, xskip):
         for y in range (ystart, yend, yskip):

            acc =0
            for i in range (num_points):
                px = contact_point_matrix[i][0]
                py = contact_point_matrix[i][1]
                nx = normals_matrix[i][0]
                ny = normals_matrix[i][1]
                acc = acc + get_point_side_of_line(x, y, px, py, nx, ny)

            if(acc == num_points):
               #on the right side of all lines - add to plot
               cor_polycone_found=True
               plt.scatter(x,y, marker='.', color="red")                     

            if(acc == -num_points):
               #on the left side of all normal lines - add to plot
               plt.scatter(x,y, marker='.', color="grey")                     
               cor_polycone_found=True
 
    if(cor_polycone_found):
        print("At least one open COR region found!")

#create and export a plot of the form closure scenario
def export_plot(poly_points, contact_points, normals, file_path):

    #plot_polygon_points(poly_points)
    fig, ax = plt.subplots() 
    ax.set_xlim(left=-20,right=150)
    ax.set_ylim(bottom=-20,top=150)
    plot_polygon_points(poly_points)
    plot_cor_regions(contact_points, normals)
    plot_contact_points(contact_points)
    plot_normals(contact_points, normals)

    plt.savefig(file_path)


#--------------------- 
# File utilities
#-----------------

#loads the polygon points from a csv file
def load_and_parse_polygon_csv(polygon_file_path):
    poly_points=np.empty((0,2),float)
       
    with open (polygon_file_path) as csv_file:
        csv_reader = csv.reader( filter(lambda row: row[0]!='#', csv_file ) )
        for row in csv_reader:
                px = float(row[0])
                py = float(row[1])
                point = np.array([[px,py]])
                poly_points=np.append(poly_points, point, axis=0)

    return poly_points


#loads the contact points and normal angle from a csv file
def load_and_parse_contact_csv(contacts_file_path):
    contact_points=np.empty((0,2),float)
    normals=np.empty((0,2),float)
       
    with open (contacts_file_path) as csv_file:
        csv_reader = csv.reader( filter(lambda row: row[0]!='#', csv_file ) )
        for row in csv_reader:
                cx = float(row[0])
                cy = float(row[1])
                point = np.array([[cx,cy]])
                contact_points=np.append(contact_points, point, axis=0)
                radangle = math.radians(float(row[2]))
                nx = math.cos(radangle)
                ny = math.sin(radangle)
                normal = np.array([[nx,ny]])
                normals = np.append(normals, normal, axis=0)
    return contact_points, normals


#--------------------- 
#
# Main Routine 
# 
# Form closure evaluation algorithm
#
#-----------------

#calculate  the matrix of wrenches to optimize
#from our loaded contact points and normals
def calculate_wrenches(contact_point_matrix, normal_matrix):
    num_points = len(contact_point_matrix)
    wrenches = np.empty((3,0),float)
    for i in range(num_points):
        p=contact_point_matrix[i]
        n=normal_matrix[i]
        m=np.cross(p,n)
        #the wrench is F=(mz, nx, ny)
        wrench=np.array( [ [m], [n[0]], [n[1]] ])
        wrenches = np.append(wrenches, wrench, axis=1)
    return wrenches


# use a linear programming approach to evaluate the first order form closure of 
# a planar polygon, as speficied in Modern Robotics 12.7
# returns 
#   True is the body is in form closure
#   False if the body is not in form closure
def calculate_form_closure(wrenches):

    #verify that the matrix is at full rank
    rank = nplin.matrix_rank(wrenches)
    print(f"Rank of wrench matrix is {rank}")

    if(rank < 3):
        #matrix with rank < 3 cannot have full closure
        print(f"Matrix is not full rank. Returning False.")
        return False

    #perform the linear programming optimization

    f=np.array( [1,1,1,1] ) 
    A=np.array( [[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1]  ])
    b=np.array(([-1,-1,-1,-1]))
    Aeq=wrenches
    Beq=np.array([0,0,0])
    result = scipy.optimize.linprog(c=f,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=Beq)


    if(result.success):
        #Result found. we can achieve full closure    
        print(" Full closure is TRUE")
        return True
    else:
        #result could not be achieved - no full closure
        print(" Full closure is FALSE")
        return False    


if __name__=="__main__":


    #open case
    contacts_file = "../contacts_open.csv"
    plot_file = "../results/contacts_open.png"

    print(f"Evaluating form closure for {contacts_file}")
    print()
    poly_points=load_and_parse_polygon_csv("../polygon.csv")
    contact_points, normals = load_and_parse_contact_csv(contacts_file)
    wrenches = calculate_wrenches(contact_points, normals)
    print("Wrench Matrix: ")
    print(wrenches)
    export_plot(poly_points, contact_points, normals, plot_file)
    result = calculate_form_closure(wrenches)
    print()



    #closed case
    contacts_file = "../contacts_closed.csv"
    plot_file = "../results/contacts_closed.png"

    print(f"Evaluating form closure for {contacts_file}")
    print()
    poly_points=load_and_parse_polygon_csv("../polygon.csv")
    contact_points, normals = load_and_parse_contact_csv(contacts_file)
    wrenches = calculate_wrenches(contact_points, normals)
    print("Wrench Matrix: ")
    print(wrenches)
    export_plot(poly_points, contact_points, normals, plot_file)
    result = calculate_form_closure(wrenches)
    print()

