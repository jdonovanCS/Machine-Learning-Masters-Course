#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:01:20 2017

@author: rditljtd
"""
import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt
from collections import namedtuple
from math import sqrt
import random
try:
    import Image
except ImportError:
    from PIL import Image

#make array of RGB values for each pixel in image    
A = misc.imread('b_small.tiff', mode='RGB')

plt.imshow(A)

num_of_centroids = 16

#instatiate array of random 16 pixels (used for cluster centroids)
random_16 = []

#loop 16 times to actually choose the random 16 pixels for cluster centroids
for i in range (0, num_of_centroids):
    
    #generate random x-value
    random_x = random.randint(0, 253)
    #generate random y-value
    random_y = random.randint(0, 161)
    
    #add random pixel to array of 16 random centroids
    random_16.append(A[random_y][random_x])

#set number of iterations
iterations = 50


#print out the centroids
#print random_16

#print out the last pixel in the image
print A[161][253]

#create an array for all pixels that is equal to the array of RGB values for the image
all_pixels = A

#create an array for associating pixels to centroids
pixels_to_centroids = []

#loop the number of iteraions
for a in range(0, iterations):
    print a+1,
    
    #loop through all y-values 
    for i in range (0, len(A)):
        
        #loop through all x-values
        for j in range (0, len(A[0])):
            
            #print out this pixel's RGB values
            #print A[i][j]
            
            #set this_pixel variable equal to this pixel's RGB values
            this_pixel = A[i][j]
            
            #set distance = to maximum distance possible
            distance = 3*253
            
            #create a variable for closest centroid
            pixel_to_centroid = [0, 0, 0, [0, 0, 0]]
            
            #loop through the cluster centroids
            for k in range(0, len(random_16)):
                
                #set this_distance equal to 0
                this_distance = 0
                
                #loop through the red, green, blue values for this centroid 
                for l in range (0, len(random_16[0])):
                    
                    #calculate the distance between the r, g, or b value for this centroid and this pixel
                    this_distance += abs(int(random_16[k][l]) - int(this_pixel[l]))
                
                #if the distance between the RGB values for this centroid is less than all other centroids
                if (this_distance < distance):
                    
                    #set the distance to the minimum distance calculated
                    distance = this_distance
                    
                    #set the closest centroid
                    pixel_to_centroid = [int(k), int(i), int(j), A[i][j]]
                    
            #set this pixel's centroid to closest
            pixels_to_centroids.append(pixel_to_centroid)
            
                    #all_pixels[i][j] = this_pixel
    
    
    #create variable for number of values associated with each centroid
    num_assoc_16 = []
    
    #create variable for sum of values associated with each centroid
    sum_16 = []
    
    #create variable for new centroids
    average_16 = []
    
    #loop through centroids                
    for i in range (0, len(random_16)):
        num_assoc_16.append(0)
        sum_16.append([0, 0, 0])
            
        #loop through each pixel
        for j in range(0, len(pixels_to_centroids)):
            
            #if this pixel is associated with this cluster
            if (pixels_to_centroids[j][0] == i):
                
                #number of points associated with this centroid
                num_assoc_16[i] += 1
                #sum up the R values
                sum_16[i][0] += int(pixels_to_centroids[j][3][0])
                #sum up the G values
                sum_16[i][1] += int(pixels_to_centroids[j][3][1])
                #sum up the B values
                sum_16[i][2] += int(pixels_to_centroids[j][3][2])
                
    for i in range (0, len(sum_16)):
        average_16.append([sum_16[i][0] / num_assoc_16[i],
                           sum_16[i][1] / num_assoc_16[i],
                           sum_16[i][2] / num_assoc_16[i]])
            
    #set centroids as the averages
    if (np.array_equal(np.array(random_16), np.array(average_16))):
        break
    random_16 = average_16
                
B = A

#loop through all pixels
for a in range (0, len(pixels_to_centroids)):
    
    i=pixels_to_centroids[a][1]
    j=pixels_to_centroids[a][2]
    
    k=pixels_to_centroids[a][0]
    B[i][j] = random_16[k]
    
plt.imshow(B)
plt.savefig('kmeans-' + str(iterations) + '.png')
    
#loop through y-values
#for i in range (0, len(all_pixels)):
    
    #loop through x-values
    #for j in range (0, len(all_pixels[0])):
        
        #loop through centroids
        #for k in range (0, len(random_16)):
            
            #if this pixel is closest to this centroid
            #if all_pixels[i][j] = random_16[k]
    

