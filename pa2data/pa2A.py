#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:32:55 2017

@author: rditljtd
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from map_features import *

x = np.loadtxt('ax.dat')
y = np.loadtxt('ay.dat')

#Plot original data out
plt.scatter(x, y, facecolors='none', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#Get the number of examples
m = x.shape[0]
#Reshape x to be a 2D column vector
x.shape = (m,1)
#Add a column of ones to x
X = np.hstack([np.ones((m,1)), x, x**2, x**3, x**4, x**5])

theta = np.zeros(shape=(6,1))

lambdaa = [0, 1, 10, -1, 5, 12]

#L2norm = np.linalg.norm(X)

#print L2norm

for i in range(len(lambdaa)):
    
    matrix = np.diag([0, 1,1,1,1,1]) 
    #print matrix

    theta = np.linalg.inv(X.transpose().dot(X) + lambdaa[i]*(matrix)).dot(X.transpose().dot(y))
    print "lambda: " + `lambdaa[i]`
    print "theta: " + `theta`
    print "L2Norm: " + `np.linalg.norm(theta)`
    #Plot original data out    
    plt.plot(X[:,1],np.dot(X,theta))
    plt.legend(['Linear Regression', 'Training Data'])
    plt.scatter(x, y, facecolors='none', color='red')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.show()
    i = i+1
    
#def computeCost(X, y, theta):
#    m=y.size
#    estimates = (X-y[:,None]).dot(theta).flatten()
#    squared_errors = (estimates) ** 2
#    sum_errors = (1 / (2 * m))*squared_errors.sum()
#    J = lambdaa*(theta)
#    return J
#    
#    #sum_first = ((hypothesis(X)-y)**2).sum()
#    #sum_second = lambdaa*((theta**2).sum)
#    #J_theta = (1/(2*m))*(sum_first+sum_second)
#    
#def gradientDescent(X, y, theta, lambdaa, iterations):
#    m = y.size
#    J_history = np.zeros(shape = (iterations, 1))
#    
#    for i in range(iterations):
#        J_history[i,0] = computeCost(X, y, theta)
#        
#    return theta, J_history
#    
##theta, J_history = gradientDescent(X, y, theta, lambdaa, 10)
#
#print theta
    