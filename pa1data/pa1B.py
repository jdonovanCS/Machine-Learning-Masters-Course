#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:23:00 2017

@author: rditljtd
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = np.loadtxt('bx.dat')
y = np.loadtxt('by.dat')


test = [1650, 3]

#Get the number of examples
m = x.shape[0]

#Reshape x to be a 2D column vector
x.shape = (m,2)

sigma = np.std(x,axis=0) #std
mu = np.mean(x,axis=0) #mean
x = (x-mu) / sigma #adjustment

test = (test-mu) / sigma

#Add a column of ones to x
X = np.hstack([np.ones((m,1)), x])

#initialize theta
theta = np.zeros(shape=(3,1)) #Initialize theta
#print theta
alpha = 1.#Your learning rate#
#J = []
iterations = 50
J_history = np.zeros(shape = (iterations, 1))

#Closed-form solution

theta2 = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
print theta2

predict2 = np.array([1, test[0], test[1]]).dot(theta2).flatten()
print 'For a house with 3 bedrooms and 1650 sqft, we predict a price of %f' % (predict2)


#gradient descent solution

def computeCost(X, y, theta):
    m = y.size
    estimates = X.dot(theta).flatten()
    squaredErrors = (estimates - y) ** 2
    J = (1.0 / (2 * m)) * squaredErrors.sum()
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m=y.size
    
    estimates = X.dot(theta).flatten()
    err_x1 = (estimates - y) * X[:, 0]
    err_x2 = (estimates - y) * X[:, 1]
    err_x3 = (estimates - y) * X[:, 2]
        #print "err_x1: " + `err_x1` + " err_x2: " + `err_x2` + " err_x3: " + `err_x3`
        
    theta[0][0] = theta[0][0] - alpha * (1.0/m) * err_x1.sum()
    theta[1][0] = theta[1][0] - alpha * (1.0/m) * err_x2.sum()
    theta[2][0] = theta[2][0] - alpha * (1.0/m) * err_x3.sum()
        
    
        
    return theta

for i in range(iterations):
    theta = gradientDescent(X, y, theta, alpha, iterations)#might need to take i out
    J_history[i] = computeCost(X, y, theta)
    
#Now plot J
plt.plot(range(iterations), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

print theta

print np.array([1, test[0], test[1]]).dot(theta)
predict1 = np.array([1, test[0], test[1]]).dot(theta).flatten()
print 'For a house with 3 bedrooms and 1650 sqft, we predict a price of %f' % (predict1)