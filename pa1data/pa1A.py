# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = np.loadtxt('ax.dat')
y = np.loadtxt('ay.dat')

#Plot original data out
plt.scatter(x, y, facecolors='none', color='red')
plt.xlabel('Age in years')
plt.ylabel('Height in meters')
plt.show()

#Get the number of examples
m = x.shape[0]
#Reshape x to be a 2D column vector
x.shape = (m,1)
#Add a column of ones to x
X = np.hstack([np.ones((m,1)), x])


#initialize theta
theta = np.zeros(shape=(2,1))


#gradient descent
alpha = .07
loops = 1300

def computeCost(X, y, theta):
    m = y.size
    estimates = X.dot(theta).flatten()
    squaredErrors = (estimates - y) ** 2
    J = (1.0 / (2 * m)) * squaredErrors.sum()
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m=y.size
    J_history = np.zeros(shape = (iterations, 1))
    
    for i in range(iterations):
        estimates = X.dot(theta).flatten()
        err_x1 = (estimates - y) * X[:, 0]
        err_x2 = (estimates - y) * X[:, 1]
        
        theta[0][0] = theta[0][0] - alpha * (1.0/m) * err_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0/m) * err_x2.sum()
        
        J_history[i,0] = computeCost(X, y, theta)
        
    return theta, J_history


theta, J_history = gradientDescent(X, y, theta, alpha, 1300)

print theta

predict1 = np.array([1, 3.5]).dot(theta).flatten()
print 'For age 3.5, we predict a height of %f' % (predict1)
predict2 = np.array([1, 7]).dot(theta).flatten()
print 'For age 7, we predict a height of %f' % (predict2)


plt.plot(X[:,1],np.dot(X,theta))
plt.legend(['Linear Regression', 'Training Data'])
plt.scatter(x, y, facecolors='none', color='red')
plt.xlabel('Age in years')
plt.ylabel('Height in meters')
plt.show()



#Display Surface Plot of J
t0 = np.linspace(-3,3,100)
t1 = np.linspace(-1,1,100)
t0.shape = (len(t0),1)
t1.shape = (len(t1),1)
T0, T1 = np.meshgrid(t0,t1)
J_vals = np.zeros((len(t0),len(t1)))
for i in range(len(t0)):
    for j in range(len(t1)):
        t = np.hstack([t0[i], t1[j]])
        J_vals[i,j] = (X, y, t)

#Because of the way meshgrids work with plotting surfaces
#we need to transpose J to show it correctly
J_vals = J_vals.T
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(T0,T1,J_vals)
plt.show()
plt.close()
#Display Contour Plot of J
plt.contour(T0,T1,J_vals, np.logspace(-2,2,15))
plt.show()