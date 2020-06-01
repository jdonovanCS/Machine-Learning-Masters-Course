#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:23:00 2017

@author: rditljtd
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.special import expit
from map_features import *

x = np.loadtxt('bx.dat', delimiter=",")
y = np.loadtxt('by.dat')


##Get the number of examples
#m = x.shape[0]
#
##Reshape x to be a 2D column vector
#x.shape = (m,2)
#
##Add a column of ones to x
#X = np.hstack([np.ones((m,1)), x])
#
##initialize theta
theta = np.zeros(shape=(28,1)) #Initialize theta
##print theta
##alpha = .3#Your learning rate#
##J = []
iterations = 15
J_history = np.zeros(shape = (iterations, 1))

#pos = np.nonzero(y)
#neg = np.where(y==0)[0]
#
#plt.scatter(x[pos,0],x[pos,1],marker='+')
#plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o', color='r')
#plt.show()

#Find indices of positive and negative examples
pos = np.nonzero(y)
neg = np.where(y==0)[0]
#Plot out the data
plt.scatter(x[pos,0],x[pos,1],marker='+')
plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o',color='r')
plt.axis('equal')
plt.show()

X=map_features(x[:,0], x[:,1])
#Newtons method solution

def sigmoid(z):
    # print "z: " + `z`
    toreturn = expit(z)
    # print "sigmoid: " + `toreturn`
    return toreturn
    
def hypothesis(X, theta):
    # print "here: " + `theta`
    # print "there: " + `X`
    toreturn = (sigmoid(np.dot(X, theta)))
    # print "hypothesis: " + `toreturn`
    return toreturn

def computeCost(X, y, theta, lambdaa):
    m = y.size
    firstOp = ((1.0/m) * (-y*np.log(hypothesis(X, theta))) - (1.0-y)*(np.log(1.0-hypothesis(X, theta)))).sum()
    toreturn = ((1.0*lambdaa/(2.0*m)) * (theta**2.0)).sum() - firstOp
    #print "computeCost: " + `toreturn`
    return toreturn

def gradientDescent(X, y, theta, lambdaa):
    m = y.size
    minus = hypothesis(X, theta) - y
    result = (1.0/m)*(X.T.dot(minus))
    toreturn = ((lambdaa*1.0)/m) * theta + (result)
    #print "gradientDescent: " + `toreturn`
    #print "result: " + `result`
    return toreturn

def hessian(X, y, theta, lambdaa):
    m = y.size
    minus = (1.0-hypothesis(X,theta))*(hypothesis(X, theta))
    Xm = minus*X
    result = (1.0/m)*(Xm.T.dot(X))
    matrix = np.diag([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    toreturn = ((lambdaa*1.0)/m) * matrix + (result)
    # print "hessian: " + `toreturn`
    #result and toreturn are the same?
    return toreturn

def newtonsMethod(X, y, theta, lambdaa, iterations):
    m=y.size
    y.shape = (m,1)
    for i in range(iterations):
        J_history[i] = computeCost(X, y, theta, lambdaa)
        theta = (theta - np.linalg.inv((hessian(X, y, theta, lambdaa))).dot(gradientDescent(X, y, theta, lambdaa)))
    #print theta, J_history
    return theta, J_history
    
def runNewtonsMethod(l):
    lambdaa = l
    theta1, J_history = newtonsMethod(X, y, theta, lambdaa, 15)    
    print 'Final theta: ', theta1
    print 'Lambda: ' + `lambdaa` + ' L2Norm: ', np.linalg.norm(theta1)
    
    #Define the ranges of the grid
    u = np.linspace(-1,1.5,200)
    v = np.linspace(-1,1.5,200)
    #Reshape to be 2-D
    u.shape = (len(u),1)
    v.shape = (len(v),1)
    #Plotting commands below
    X1, Y = np.meshgrid(u,v)
    Z = np.zeros((len(u),len(v)))
    #Initialize z = theta*x over the whole grid
    for i in range(len(u)):
        for j in range(len(v)):
            Z[j][i] = np.dot(map_features(u[i],v[j]),theta1)
    plt.clf()
    plt.scatter(x[pos,0],x[pos,1],marker='+')
    plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o',color='r')
    plt.axis('equal')
    plt.contour(X1,Y,Z, 0,linewidth=2)
    plt.show()  
    
i=0
lambdaas=[0, 1, 10, -2, 4, 12]
for i in range(len(lambdaas)):
    runNewtonsMethod(lambdaas[i])