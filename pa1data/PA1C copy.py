#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:36:53 2017

@author: rditljtd
"""

#Number of examples
m = x.shape[0]
#y must be reshaped to work
y.shape = (y.shape[0],1)

#Initialize theta
#It needs to be a 2D array now
theta = np.zeros((x.shape[1],1))
#What are our predictions?
h = np.dot(x.T,theta)
#Compute gradient
# If y is 1D, h-y will be m x m (isn't numpy fun!)
gradient = np.dot(x.T,h-y)/m
#Update theta with the computed gradient
theta = theta - (alpha/m)*gradient

#Compute theta using normal equations
xTx = np.dot(x.T,x)
xTxi = np.linalg.inv(xTx)
xTxixT = np.dot(xTxi,x.T)
theta = np.dot(xTxixT,y)

#Reshape y, so that things work out
y.shape = (y.shape[0],1)
#theta needs to be 2D
theta = np.zeros((x.shape[1],1))
#Compute our vector of predictions
h = expit(np.dot(x,theta))
#Compute our cost (mean of per-example cost)
J = ((-y * np.log(h)) - ((1 - y)*np.log(1 - h))).mean()

H = np.zeros((x.shape[1],x.shape[1]))
for i in range(m):

    #What is our prediction for example i?
    # h is a scalar
    h = expit(np.dot(theta, x[i,:]))
    #Compute Hessian H (broken into bitesize steps)
    nH1 = h*(1-h)*x[i,:]

    #Reshape components so that nH3 is calculated
    # correctly
    # We have to do this because x[i,:] is 1D
    nH1.shape = (len(nH1),1)
    nH2 = x[i,:]
    nH2.shape = (len(nH2),1)
    nH3 = np.dot(nH1,nH2.T)
    #Add H from this step (nH3) to total H
    H = H + nH3

#Do final divide by m to get Hessian
H = (1/float(m))*H

#Reshape y so everything works out
y.shape = (y.shape[0],1)
#Initialize theta (must be 2D for this)
theta = np.zeros((x.shape[1],1))
#Compute prediction vector
h = expit(np.dot(x,theta))
#Compute Hessian
H = np.dot((h*(1-h)*x).T,x)/m