import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import modern_robotics as mr

def es1():
    print("ES1")
    plt.axis([-3, 5, -3, 5])
    for s in np.arange(0, 1, 0.01):
        print(s)
        x = 1 + 2 * np.cos(np.pi * s)
        y = 2 * np.sin(np.pi * s)
        print(x, y)
        plt.scatter(x, y)
    plt.show()
    print('_'*70)

def es2():
    print('ES2')
    plt.axis([-5, 5, -5, 5])
    for s in np.arange(0, 1, 0.01):
        x = 1.5 * (1 - np.cos(2 * np.pi * s))
        y =np.sin(2 * np.pi * s)
        plt.scatter(x, y)
    plt.scatter(0,0, c='r')
    plt.scatter(1.5, 1, c='r')
    plt.scatter(3,0, c='r')
    plt.scatter(1.5, -1, c='r')
    plt.show()
    print('_'*70)

def es4():
    print('ES4')
    t = sym.symbols('t')
    T = sym.symbols('T')
    a_0 = 0
    a_1 = 0
    a_2 = 0
    a_3 = 10 / (T ** 3)
    a_4 = -15 / (T ** 4)
    a_5 = 6 / (T ** 5)
    s = a_0 + a_1 * t + a_2 * t ** 2 + a_3 * t ** 3 + a_4 * t ** 4 + a_5 * t ** 5
    s_dot = sym.Derivative(s, t).doit()
    s_dotdot = sym.Derivative(s_dot, t).doit()
    assert (s.subs(t, T) == 1)
    assert (s.subs(t, 0) == 0)
    assert (s_dot.subs(t, 0) == 0)
    assert (s_dot.subs(t, T) == 0)
    assert (s_dotdot.subs(t, 0) == 0)
    assert (s_dotdot.subs(t, T) == 0)
    print('OK')
    print('_'*70)

def es5():
    print("ES5")
    Tf=5
    t=3
    s = mr.QuinticTimeScaling(Tf, t)
    print(f"s: {s}")
    print('_'*70)

def es6():
    N = 10
    X_start = np.eye(4)
    X_end = np.array([[0, 0, 1, 1],
                      [1, 0, 0, 2],
                      [0, 1, 0, 3],
                      [0, 0, 0, 1]])
    Tf = 10 #duration of trajectory
    method = 3 #cubic time scaling
    cubic_screw_trajectory = mr.ScrewTrajectory(X_start, X_end, Tf, N, method)
    print(f'the {len(cubic_screw_trajectory)} matrices of the Screw trajectory: \n{cubic_screw_trajectory}')
    print('_'*70)

def es7():
    Tf = 10 #duration of trajectory
    N = 10 #ten matrices
    method = 5 #quintic time scaling
    X_start = np.eye(4)
    X_end = np.array([[0, 0, 1, 1],
                      [1, 0, 0, 2],
                      [0, 1, 0, 3],
                      [0, 0, 0, 1]])
    quintic_cartesian_trajectory = mr.CartesianTrajectory(X_start, X_end, Tf, N, method)
    print(f'the {len(quintic_cartesian_trajectory)} matrices of the Cartesian trajectory: \n{quintic_cartesian_trajectory}')
    print('_' * 70)

#-----------------------
if __name__=='__main__':
    es1()
    es2()
    es4()
    es5()
    es6()
    es7()