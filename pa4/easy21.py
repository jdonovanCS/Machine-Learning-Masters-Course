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

hit = False
currentPlayerScore = 0
currentDealerScore = 0
initialDealerScore = 0
bust = False
playGame = True
win = False
possibleDealerInitialScores = [1,2,3,4,5,6,7,8,9,10]
possiblePlayerScores = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
possibleHitValues = [True, False]
states = {}
for dis in possibleDealerInitialScores:
    for ps in possiblePlayerScores:
        for hv in possibleHitValues:
            states[dis, ps, hv] = 0
#states = [possibleDealerInitialScores, possiblePlayerScores, possibleHitValues] 
#[DealerInitialScore, PlayerCurrentScore, hit]
reward = states
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
    #print
    #print ("Dealer first card: ")
    global currentDealerScore
    global initialDealerScore
    currentDealerScore = getCard(True)
    initialDealerScore = currentDealerScore
    #print ("Score: " + str(currentDealerScore))
    #print
    #print "--------------PLAYER---------------------"
    #print
    #print ("Player first card: ")
    global currentPlayerScore
    currentPlayerScore = getCard(True)
    #print ("Score: " + str(currentPlayerScore))

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

#For every action taken during a game, if the game ends in a win add 1 to that state.

while playGame == True:
    initializeGame()
    #hit = raw_input("Hit: Y/N\t")[0].upper()
    hit = random.choice(["Y", "N"])
    stepsTaken.append([initialDealerScore, currentPlayerScore, hit=="Y"])
    while hit == "Y":
        global currentPlayerScore
        global currentDealerScore
        global initialDealerScore
        currentPlayerScore = sum([currentPlayerScore, getCard()])
        #print "Score: " + str(currentPlayerScore)
        if currentPlayerScore > 21 or currentPlayerScore < 1:
            #print "BUST"
            bust = True
            hit = "N"
            win = False
            continue
        #hit = raw_input("Hit: Y/N\t")[0].upper()
        hit = random.choice(["Y", "N"])
        stepsTaken.append([initialDealerScore, currentPlayerScore, hit=="Y"])
    
    
    #print "-----------------------DEALER-------------------------"
    #print "Score: " + str(currentDealerScore)
    while currentDealerScore < 17 and currentPlayerScore in range (1, 22) and currentDealerScore in range(1,22):
        currentDealerScore = sum([currentDealerScore, getCard()])
        #print "Score: " + str(currentDealerScore)
        if currentDealerScore > 21 or currentDealerScore < 1:
            #print "BUST"
            win = True
            continue
        #time.sleep(2)
    
    
    if win == False and currentPlayerScore in range (currentDealerScore, 22):
        win = True
    if (win == True):
        #print stepsTaken
        for step in stepsTaken:
            #print step
            reward[step[0], step[1], step[2]] += 1
        #print "YOU WON!"
    if (win == False):
        for step in stepsTaken:
            #print step
            reward[step[0], step[1], step[2]] += -1
        #print "YOU LOST!"
    endGame()
    
    if (gamesPlayed == 1000000):
        playGame = False
    if (gamesPlayed < 1000000):
        playGame = True#input("Play Again: Y/N\t")
        if (gamesPlayed % 10000 == 0):
            #print (str(gamesPlayed / 10000) + " percent complete ", end='\r')
            sys.stdout.write("\r" + str(gamesPlayed/10000) + " percent complete")
            sys.stdout.flush()
        gamesPlayed += 1
    
print ("Thanks for Playing!")
print ('Dealer Initial Value | Player Value | Hit Value == Reward Value')
sorted_reward = sorted(reward.items(), key=operator.itemgetter(1))
for indReward in sorted_reward:
    print (str(indReward[0]) + " = " + str(reward[indReward[0]]))
    
    
#Gt = x + x2(v) + x3(v^2) + x4(v^3) ...
#Ss = Ss + Gt
#Ns = Ns + 1
#Vs = Ss/Ns
##I'm confused as to how to implement the time-step to give correct reward values.
#I was not able to generate a graph or visualization


    
