#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:21:17 2017

@author: rditljtd
"""

import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt
import math
#find probability of email being spam given that it contains a particular word
#this equals the probability of seeing this word given that it is a spam * probability of it being seen overall / probability of it being spam

#find probability of email being non-spam given that it contains a particular word
#this equls the probabiltiy of seeing this word givent that the email is non-spam *
    #probability of seeing this word overall
    #/probability of email being non-spam

# Load the labels for the training set
train_labels = np.loadtxt('train-labels-50.txt',dtype=int)

# Get the number of training examples from the number of labels
numTrainDocs = train_labels.shape[0]

# This is how many words we have in our dictionary
numTokens = 2500

# Load the training set feature information
M = np.loadtxt('train-features-50.txt',dtype=int)

# Create matrix of training data
train_matrix = sps.csr_matrix((M[:,2], (M[:,0], M[:,1])),shape=(numTrainDocs,numTokens))

#train_labels[i] = ith document label: spam or not spam

#tran_matrix[i:] = ith document

#train_matrix[i:, j] = jth word in ith document

#print train_matrix[69]

print train_matrix.shape[1]

spam_prob = (sum(x == 1 for x in train_labels))/(numTrainDocs*(1.0))

nonspam_prob = (sum(x == 0 for x in train_labels))/(numTrainDocs*(1.0))

print spam_prob, nonspam_prob

#create variable for sum of all non spam emails
num_nonspam = (sum(x==0 for x in train_labels))
#create variable for sum of all spam emails
num_spam = (sum(x==1 for x in train_labels))
    
#create array for probability all words in spam and non spam [spam_prob, nonspam_prob]
prob_for_each_word = []

    
#add up the instances of this word [spam, nonspam]
sum_for_this_word = [0, 0]

#create array for how many times each word was foud [[spam, nonspam]] i = word
found_for_all_emails = []

#loop through each email
for j in range (0, train_matrix.shape[0]):
        
    #array of words in email [instances]
    words_in_email = train_matrix[j].toarray()[0]
    #print train_matrix[j],
        
    
        
    #loop through each word in dictionary and see if it is in email
    for k in range (0, len(words_in_email)):
        
        if j == 0:
            #create array for if this word was found [spam, nonspam]
            found_for_all_emails.append([0, 0])
                        
        #if email is non spam and there is more than zero instances of this word
        if (train_labels[j] == 0 and words_in_email[k] > 0):
                
            #increment the sum for this word for non spam emails
            found_for_all_emails[k][1] += 1
             
        #else if the email is spam and there is more than zero instances of this word
        elif (train_labels[j] == 1 and words_in_email[k] > 0):
                
            #print words_in_email[k]
            #increment the sum for this word for spam emails
            found_for_all_emails[k][0] += 1
         
#print found_for_all_emails
#sum_for_this_word = sum(found_for_all_emails[:1]) 
#add the probability for this word to array

#create an array to hold the probability for each word [spam, nonspam] i = word
prob_for_each_word = []

#create array to hold the overall probability for each word [num] i = word
overall_prob_for_each_word = []

for i in range (0, numTokens):
    
    prob_for_each_word.append([0,0])
    overall_prob_for_each_word.append(0)
    
    #divide by the total number of spam emails
    prob_for_each_word[i][0] = (found_for_all_emails[i][0]/(num_spam*(1.0)))
    
    #divide by the total number of nonspam emails
    prob_for_each_word[i][1] = (found_for_all_emails[i][1]/(num_nonspam*(1.0)))
    
    overall_prob_for_each_word[i] = (found_for_all_emails[i][0] + found_for_all_emails[i][1])/((num_nonspam + num_spam)*(1.0))
    

#print prob_for_each_word
#print overall_prob_for_each_word
    


# Load the labels for the training set
test_labels = np.loadtxt('test-labels.txt',dtype=int)

# Get the number of training examples from the number of labels
numTestDocs = test_labels.shape[0]

# Load the training set feature information
N = np.loadtxt('test-features.txt',dtype=int)

# Create matrix of training data
test_matrix = sps.csr_matrix((N[:,2], (N[:,0], N[:,1])),shape=(numTestDocs,numTokens))

prob_for_all_test_emails = []

#iterate through each email in the test set
for i in range (0, (test_matrix.shape[0])):
    
    #array of words in email [instances]
    words_in_test_email = test_matrix[i].toarray()[0]
    #print train_matrix[j],
    
    prob_spam_given_all_words = 0
    prob_nonspam_given_all_words = 0
    
    #loop through each word in dictionary and see if it is in email
    for k in range (0, len(words_in_test_email)):
        
        prob_spam_given_word = 0
        prob_nonspam_given_word = 0
        
        #if email is non spam and there is more than zero instances of this word
        if (words_in_test_email[k] > 0):
            #prob_nonspam_given_word = float(((prob_for_each_word[k][1])*(1.0))*overall_prob_for_each_word[k])/((nonspam_prob)*1.0)
            #prob_spam_given_word = float(((prob_for_each_word[k][0])*(1.0))*overall_prob_for_each_word[k])/((spam_prob)*1.0)
            if (prob_for_each_word[k][1] > 0):
                prob_nonspam_given_word = float(math.exp(math.log(prob_for_each_word[k][1]))) + float(math.exp(math.log(overall_prob_for_each_word[k]))) - float(math.exp(math.log(nonspam_prob)))
            if (prob_for_each_word[k][0] > 0):
                prob_spam_given_word = float(math.exp(math.log(prob_for_each_word[k][0]))) + float(math.exp(math.log(overall_prob_for_each_word[k]))) - float(math.exp(math.log(spam_prob)))
            
        if (prob_spam_given_word > 0):
            #print prob_spam_given_all_words,
            #prob_spam_given_all_words += float((prob_spam_given_word*(1.0)))
            #print prob_spam_given_all_words
            prob_spam_given_all_words += prob_spam_given_word
        
        if (prob_nonspam_given_word > 0):
            #print prob_nonspam_given_all_words,
            #prob_nonspam_given_all_words += float((prob_nonspam_given_word*(1.0)))
            prob_nonspam_given_all_words += prob_nonspam_given_word

    #prob_spam_given_all_words += float(math.exp(math.log(spam_prob)))
    #prob_nonspam_given_all_words += float(math.exp(math.log(nonspam_prob)))
    prob_for_all_test_emails.append([prob_spam_given_all_words, prob_nonspam_given_all_words])
    
print prob_for_all_test_emails
    
classify_each_test_email = []
spam_count = 0
nonspam_count = 0
incorrect = []
print len(prob_for_all_test_emails)
for k in range(0, len(prob_for_all_test_emails)):
    if prob_for_all_test_emails[k][0] > prob_for_all_test_emails[k][1]:
        classify_each_test_email.append(1)
        spam_count += 1
        if(test_labels[k] != classify_each_test_email[k]):
            incorrect.append([k, classify_each_test_email[k], test_labels[k]])
    elif prob_for_all_test_emails[k][1] > prob_for_all_test_emails[k][0]:
        classify_each_test_email.append(0)
        nonspam_count += 1
        if(test_labels[k] != classify_each_test_email[k]):
            incorrect.append([k, classify_each_test_email[k], test_labels[k]])
    else:
        classify_each_test_email.append('unknown') 
        incorrect.append([k, classify_each_test_email[k], test_labels[k]])
    
print classify_each_test_email
print "spam count: " + str(spam_count) + " spam percentage: " + str(float((spam_count*100)/(len(classify_each_test_email))))
print "nonspam count: " + str(nonspam_count) + " nonspam percentage: " + str(float((nonspam_count*100)/(len(classify_each_test_email))))
print "number misclassified: " + str(len(incorrect)) + " percentage misclassified: " + str(float(((len(incorrect))*100)/(len(classify_each_test_email))))
            
        
        
    

#def nBayes(train_matrix, train_labels):
#    print 'called nBayes'
#    #reset num_unique_words_in_spam ratio and wordsInSpam
#    
#    #create variable for number of unique words in spam emails
#    num_unique_words_in_spam = 0
#    
#    #create variable for number of unique words in nonspam emails
#    num_unique_words_in_nonspam = 0
#    
#    #total number of words in spam emails
#    totalWordsInSpam = 0
#    totalWordsInNonSpam = 0
#    wordsInSpam = {}
#    wordsInNonSpam = {}
#    
#    #loop through each email
#    for i in range (0, train_matrix.shape[0]):
#        #get array with words and # of instance of these words
#        #two dim array with first being word id and second being # of instances
#        wordsInEmail = train_matrix[i].toarray()[0]
#        #print wordsInEmail
#        #print "-------------"
#        
#        #for each possible word in the email
#        for j in range(0, len(wordsInEmail)):
#    
#            if (wordsInEmail[j] == 0):
#                continue
#            #if the email is a spam email
#            if train_labels[i] == 1:
#            
#                #increment summation value for a unique word in this spam email
#                num_unique_words_in_spam = num_unique_words_in_spam+1
#                #add the # of instances of this word to the number of words total in the email
#                totalWordsInSpam = totalWordsInSpam + wordsInEmail[j]
#                
#                #if this word has already been seen in a spam email
#                if (j in wordsInSpam.keys()):
#                    #add the # of instances of this word to the # of instances of this word in all spam emails
#                    wordsInSpam[j] += wordsInEmail[j]
#                    
#                #word has not been seen in a spam email yet    
#                else:
#                    #set the # of instances of this word in a spam email to the # of instances in this email
#                    wordsInSpam[j] = wordsInEmail[j]
#            
#            #if email is non-spam        
#            if train_labels[i] == 0:
#                #increment summation value for a unique word in this non spam email
#                num_unique_words_in_nonspam = num_unique_words_in_nonspam+1
#                #add the # of instances of this word to the number of words total in the email
#                totalWordsInNonSpam = totalWordsInNonSpam + wordsInEmail[j]
#                
#                #if this word has already been seen in a non spam email
#                if (j in wordsInNonSpam.keys()):
#                    #add the # of instances of this word to the # of instances of this word in all non spam emails
#                    wordsInNonSpam[j] += wordsInEmail[j]
#                    
#                #word has not been seen in a non spam email    
#                else:
#                    #set the # of instances of this word in non spam emails to the # of instances in this email
#                    wordsInNonSpam[j] = wordsInEmail[j]
#    
#    #I think what actually needs to be done here is that for each unique word in the dictionary, calculate the percentage of spam emails that contain that word
#    #then calculate the # of non spam emails that contain that word. Multiply the percentages for every word in 
#    
#    
#    #dictionary of words and # of instances in spam emails            
#    print wordsInSpam
#    #dictionary of words and # of instances in non spam emails
#    print wordsInNonSpam
#    #of unique words in spam email
#    print num_unique_words_in_spam
#    #of unique words in non spam email
#    print num_unique_words_in_nonspam
#    #total # of words in spam emails
#    print totalWordsInSpam
#    #total # of words in non spam emails
#    print totalWordsInNonSpam
#    
#    #return # of unique words in spam + 1 / total # of words in spam emails + # of words in dictionary,
#    #return # of unique words in non spam +1 / total # of words in non spam emails + # of words in dictionary
#    return (num_unique_words_in_spam+1)/((totalWordsInSpam+train_matrix.shape[1])*1.0), (num_unique_words_in_nonspam+1)/((totalWordsInNonSpam+train_matrix.shape[1])*1.0)
#    
#print nBayes(train_matrix, train_labels)
