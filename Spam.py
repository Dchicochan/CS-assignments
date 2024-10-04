import pandas as pd
import os
import csv
from collections import defaultdict
import math


fileName = 'SpamDetection.csv'
filePath = os.path.join(os.getcwd(), fileName)

#reading csv file
df = pd.read_csv(fileName)
df['data'] = df['data'].str.lower()

#separating training and testing sets
print("Splitting data sets...\n")
trainList = df[:20]
testList = df[-10:]

print(f"Training set:\n {trainList}\n")
print(f"Testing set:\n {testList}\n")

#initializing spam and ham lists from training set
spamList = trainList[trainList['Target'] == 'spam']
hamList = trainList[trainList['Target'] == 'ham']
union = pd.concat([spamList, hamList]).drop_duplicates().reset_index(drop=True)#total vocab words

#initializing prior probabilities
spamProb = len(spamList)/len(trainList)
hamProb = len(hamList)/len(trainList)

print(f"Prior spam probability: {spamProb}")
print(f"Prior ham probability: {hamProb}")

#initializing word dictionary for ham and spam
spamWords = defaultdict(int)
hamWords = defaultdict(int)

#storing words to dictionaries
for message in spamList['data']:
    for word in message.split():
        spamWords[word] += 1

for message in hamList['data']:
    for word in message.split():
        hamWords[word] += 1

#initialize total words 
totalSwords = sum(spamWords.values())
totalHwords = sum(hamWords.values())

#total vocab size
totalVocab = len(spamWords) + len(hamWords)

def condProb(message, dict):#calculates conditional probability
    words = message.split()
    totalWords = sum(dict.values())
    

    totalProb = 1

    for word in words:
        wordCount = dict.get(word, 0)
        prob = (wordCount + 1)/(totalWords + totalVocab)
        totalProb *= prob
        '''
        #debug statements
        print(f"{word}, total prob: {prob}")
        print(f"condprob: {wordCount} + 1/{totalWords} + {totalVocab}\n")
        '''
        
    
    return totalProb       

def postProb(message, prob, dict):#calculates posterior probability
    if dict == spamWords:
        print(f"Conditional probability for spam {condProb(message, dict)}%")
    else:
        print(f"Conditional probability for ham {condProb(message, dict)}%")
    return prob * condProb(message, dict)


messageID = []#initializing identification list for messages

print("\nTest cases:")
for message in testList['data']:
    print(f"message:{message}")
   
    spamPost = postProb(message, spamProb, spamWords)
    hamPost = postProb(message, hamProb, hamWords)
    
    
    print(f"posterior probability of spam: {spamPost}%")
    print(f"posterior probability of ham: {hamPost}%")

    if spamPost > hamPost:
        print("marked as: SPAM\n")
        messageID.append("spam")
            
    else:
        print("marked as HAM\n")
        messageID.append("ham")

accuracy = 100* (sum(1 for msgID, dictID in zip(messageID, testList['Target']) if msgID == dictID)/len(messageID))


print(f"\nAccuracy: {accuracy}%\n")
