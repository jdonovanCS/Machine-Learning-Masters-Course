from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:07:17 2017

@author: rditljtd
"""

from random import randint
import operator
import sys
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

hit = False
currentPlayerScore = 0
currentDealerScore = 0
initialDealerScore = 0
bust = False
playGame = True
win = False

#Begin------------------------------------------------------------------------------------------------------------
#Changed: commented out portion of code instantiating states. Instead doing it as a numpy array
#possibleDealerInitialScores = [1,2,3,4,5,6,7,8,9,10]
#possiblePlayerScores = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#possibleHitValues = [True, False]
#states = {}
#for dis in possibleDealerInitialScores:
#    for ps in possiblePlayerScores:
#        for hv in possibleHitValues:
#            states[dis, ps, hv] = 0



#states = [possibleDealerInitialScores, possiblePlayerScores, possibleHitValues] 
#[DealerInitialScore, PlayerCurrentScore, hit]
reward = np.zeros((11,22,2))
states = np.zeros(reward.shape)
#End------------------------------------------------------------------------------------------------------------

stepsTaken = []
gamesPlayed = 0


def getCard(isFirstCard = False):  
    if (isFirstCard):
        color = 2
    else:
        color = randint(1, 3)
    number = randint(1,10)
    if color == 1:
        #print "red " + str(number)
        number = -number
    #if color == 2 or color == 3:
        #print "black " + str(number)
    return number



#Draw first card for both dealer and player
def initializeGame():

    global currentDealerScore
    global initialDealerScore
    currentDealerScore = getCard(True)
    initialDealerScore = currentDealerScore

    global currentPlayerScore
    currentPlayerScore = getCard(True)

#reset variables
def endGame():
    firstTimeDealer = True
    firstTimePlayer = True
    hit = "Y"
    global currentPlayerScore
    currentPlayerScore = 0
    global initialDealerScore
    initialDealerScore = 0
    global currentDealerScore
    currentDealerScore = 0
    global win
    win = False
    global stepsTaken
    stepsTaken = []

#Begin------------------------------------------------------------------------------------------------------------
#Changed: epsilonGreedy function added
def epsilonGreedy(s, Q, epsilon):
    if random.random() < epsilon:
        return random.choice(["Y", "N"])
    else:
        return "N" if Q[s[0]][s[1]][0] > Q[s[0]][s[1]][1] else "Y"
#End------------------------------------------------------------------------------------------------------------

#Begin------------------------------------------------------------------------------------------------------------
#Changed: getEpsilon Function added
def getEpsilon(state, states):
    N0 = 100
    NS = states[state[0]][state[1]][0] + states[state[0]][state[1]][1]
    return N0 / (N0 + NS)
#End------------------------------------------------------------------------------------------------------------

#Begin------------------------------------------------------------------------------------------------------------
#Changed: added a function for plotting the data
def plot_vf(Q):
    #Plot the value function represented by Q
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(list(range(11)), list(range(22)))
    ax.plot_surface(xx, yy, np.max(Q, axis=2).T)
    plt.show()
#End------------------------------------------------------------------------------------------------------------
    
    
#For every action taken during a game, if the game ends in a win add 1 to that state.

#MCControl
while playGame == True:

#Begin------------------------------------------------------------------------------------------------------------    
    #Changed: Added variables for epsilon
    global epsilon
#End------------------------------------------------------------------------------------------------------------

    initializeGame()

#Begin------------------------------------------------------------------------------------------------------------
    #Changed: instantiate epsilon
    epsilon = getEpsilon(state, states)
#End------------------------------------------------------------------------------------------------------------

#Begin------------------------------------------------------------------------------------------------------------    
    #Changed: changed the way the action is determined
    hit = epsilonGreedy(state,states,epsilon)#random.choice(["Y", "N"])
    state = (initialDealerScore, currentPlayerScore)
    if (hit=="Y"):
        action = 1
    else:
        action = 0
    stepsTaken.append((state[0], state[1], action))
#End------------------------------------------------------------------------------------------------------------
    
    while hit == "Y":
        currentPlayerScore = sum([currentPlayerScore, getCard()])
        if currentPlayerScore > 21 or currentPlayerScore < 1:
            bust = True
            hit = "N"
            win = False
            continue

#Begin------------------------------------------------------------------------------------------------------------        
        #Changed: the way the action is determined and adding state to states taken in this episode
        hit = epsilonGreedy(state,states,epsilon)#random.choice(["Y", "N"])
        if (hit == "Y"):
            action = 1
        else:
            action = 0
        state = (currentDealerScore, currentPlayerScore)
        stepsTaken.append((state[0], state[1], action))
#End------------------------------------------------------------------------------------------------------------    
    
    while currentDealerScore < 17 and currentPlayerScore in range (1, 22) and currentDealerScore in range(1,22):
        currentDealerScore = sum([currentDealerScore, getCard()])
        if currentDealerScore > 21 or currentDealerScore < 1:
            win = True
            continue
    
    
    if win == False and currentPlayerScore in range (currentDealerScore+1, 22):
        win = True

#Begin------------------------------------------------------------------------------------------------------------    
    #Changed: added in method for ties
    elif win == False and currentPlayerScore == currentDealerScore:
        win = None
    #Changed: added in reward and increment for ties
    if (win == None):
        for step in stepsTaken:
            #Changed: to increment counter of state
            states[step[0], step[1], step[2]] += 1
            #Changed: to add reward for state
            reward[step[0], step[1], step[2]] += (1/(1.0*(states[step[0]][step[1]][step[2]])) * (0 - reward[step[0]][step[1]][step[2]]))
#End------------------------------------------------------------------------------------------------------------
        
    if (win == True):
        for step in stepsTaken:

#Begin------------------------------------------------------------------------------------------------------------
            #Changed: to increment counter of state
            states[step[0]][step[1]][step[2]] += 1
#End------------------------------------------------------------------------------------------------------------

#Begin------------------------------------------------------------------------------------------------------------
            #Changed: to add reward for state
            reward[step[0]][step[1]][step[2]] += (1/(1.0*(states[step[0]][step[1]][step[2]])) * (1 - (reward[step[0], step[1], step[2]])))
#End------------------------------------------------------------------------------------------------------------

    if (win == False):
        for step in stepsTaken:

#Begin------------------------------------------------------------------------------------------------------------
            #Changed: to increment counter of state
            states[step[0]][step[1]][step[2]] += 1
#End------------------------------------------------------------------------------------------------------------
            
#Begin------------------------------------------------------------------------------------------------------------
            #Changed: to add reward for state
            reward[step[0]][step[1]][step[2]] += (1/(1.0*(states[step[0]][step[1]][step[2]])) * (-1 - (reward[step[0]][step[1]][step[2]])))
#End------------------------------------------------------------------------------------------------------------
    
    endGame()
    
    if (gamesPlayed == 1000000):#000000):
        playGame = False
    if (gamesPlayed < 1000000):#000000):
        playGame = True
        if (gamesPlayed % 10000 == 0):
            sys.stdout.write("\r" + str(gamesPlayed/10000) + " percent complete")
            sys.stdout.flush()
        gamesPlayed += 1
    
#print (reward)
plot_vf(reward)



    
