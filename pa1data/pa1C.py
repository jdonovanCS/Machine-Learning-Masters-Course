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


#Get the number of examples
m = x.shape[0]

#Reshape x to be a 2D column vector
x.shape = (m,2)

#Add a column of ones to x
X = np.hstack([np.ones((m,1)), x])

#initialize theta
theta = np.zeros(shape=(3,1)) #Initialize theta
#print theta
#alpha = .3#Your learning rate#
#J = []
iterations = 15
J_history = np.zeros(shape = (iterations, 1))

pos = np.nonzero(y)
neg = np.where(y==0)[0]

plt.scatter(x[pos,0],x[pos,1],marker='+')
plt.scatter(x[neg,0],x[neg,1],facecolors='none',marker='o', color='r')
plt.show()


#Newtons method solution

def sigmoid(z):
    print "z: " + `z`
    toreturn = expit(z)
    print "sigmoid: " + `toreturn`
    return toreturn
    
def hypothesis(X, theta):
    print "here: " + `theta`
    print "there: " + `X`
    toreturn = (sigmoid(np.transpose(theta))*(X))
    print "hypothesis: " + `toreturn`
    return toreturn


def computeCost(X, y, theta):
    m = y.size
    toreturn = ((1.0/m) * (-y.dot(np.log(hypothesis(X, theta))) - (1-y).dot(np.log(1-hypothesis(X, theta)))).sum())
    print "computeCost: " + `toreturn`
    return toreturn

def gradientDescent(X, y, theta):
    m = y.size
    minus = hypothesis(X, theta).subtract(hypothesis(X,theta), y)
    toreturn = (1.0/m)*(minus.dot(X))
    print "gradientDescent: " + `toreturn`
    return toreturn

def hessian(X, y, theta):
    m = y.size
    minus = 1-hypothesis(X,theta)*(hypothesis(X, theta))
    xTrans = (X.transpose()).dot(X)
    result = minus.dot(xTrans)
    print np.sum(result)
    toreturn = (1.0/m)*((minus).dot(xTrans)).sum()
    print "hessian: " + `toreturn`
    return toreturn

def newtonsMethod(X, y, theta, iterations):
    m=y.size
    
    for i in range(iterations):
        J_history[i] = computeCost(X, y, theta)
        theta = (theta - (1/(hessian(X, y, theta))) * gradientDescent(X, y, theta))
        
    print theta, J_history
    return theta, J_history
    
theta, J_history = newtonsMethod(X, y, theta, 15)    
    