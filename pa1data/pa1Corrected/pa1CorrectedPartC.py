#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:23:00 2017

@author: rditljtd
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.special import expit

x = np.loadtxt('cx.dat')
y = np.loadtxt('cy.dat')

#--------------------------------------------------------------------------------------------
#Changed: added test values
test = [2.0e+01, 8.0e+01]
#--------------------------------------------------------------------------------------------


#Get the number of examples
m = x.shape[0]

#Reshape x to be a 2D column vector
x.shape = (m,2)

#Add a column of ones to x
X = np.hstack([np.ones((m,1)), x])

#initialize theta
theta = np.zeros(shape=(3,1)) #Initialize theta

iterations = 13
J_history = np.zeros(shape = (iterations, 1))

pos = np.nonzero(y)
neg = np.where(y==0)[0]

plt.scatter(x[pos,0],x[pos,1],marker='+')
plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o', color='r')
plt.show()


#Newtons method solution

def sigmoid(z):
    toreturn = expit(z)
    return toreturn
    
def hypothesis(X, theta):

#--------------------------------------------------------------------------------------------
#Changed: math logic - np.transpose to np.dot
    toreturn = (sigmoid(np.dot(X, theta)))
#--------------------------------------------------------------------------------------------

    return toreturn

def computeCost(X, y, theta):
    m = y.size

#--------------------------------------------------------------------------------------------
#Changed: math logic - correct use of * instead of np.dot
    toreturn = ((1.0/m) * (-y*np.log(hypothesis(X, theta))) - (1-y)*(np.log(1-hypothesis(X, theta)))).sum()
#--------------------------------------------------------------------------------------------

    #print "computeCost: " + `toreturn`
    return toreturn

def gradientDescent(X, y, theta):
    m = y.size

#--------------------------------------------------------------------------------------------
#Changed: math logic - correct use of "-" instead of np.subtract
    minus = hypothesis(X, theta) - y
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#Changed: math logic - use of X.T.dot instea of just np.dot
    toreturn = (1.0/m)*(X.T.dot(minus))
#--------------------------------------------------------------------------------------------

    return toreturn

def hessian(X, y, theta):
    m = y.size
    minus = (1-hypothesis(X,theta))*(hypothesis(X, theta))
    
#--------------------------------------------------------------------------------------------
#Changed: math logic on multiplying minus and X and transposing
    Xm = minus*X
    result = (1.0/m)*(Xm.T.dot(X))
#--------------------------------------------------------------------------------------------

    toreturn = result
    # print "hessian: " + `toreturn`
    return toreturn

def newtonsMethod(X, y, theta, iterations):
    m=y.size
    y.shape = (m,1)
    for i in range(iterations):
        J_history[i] = computeCost(X, y, theta)

#--------------------------------------------------------------------------------------------
#Changed: used np.linalg.inv instead of 1/x and used np.dot instead of *
        theta = (theta - np.linalg.inv((hessian(X, y, theta))).dot(gradientDescent(X, y, theta)))
#--------------------------------------------------------------------------------------------

    print theta, J_history
    return theta, J_history
    
theta, J_history = newtonsMethod(X, y, theta, iterations)    
print 'Final theta: ', theta
    
plt.plot(range(iterations), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print np.array([1, test[0], test[1]]).dot(theta)
predict1 = np.array([1, test[0], test[1]]).dot(theta).flatten()
print predict1

#--------------------------------------------------------------------------------------------
#Changed: Added code to plot decision boundary
theta = theta[:,0]  # Make theta a 1-d array.
x2 = np.linspace(0,100, 80)
y2 = -(theta[0] + theta[1]*x2)/theta[2]
plt.scatter(x[pos,0],x[pos,1],marker='+')
plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o', color='r')
plt.plot(x2, y2)
plt.show
#--------------------------------------------------------------------------------------------
