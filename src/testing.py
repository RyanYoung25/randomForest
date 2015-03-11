#!/usr/bin/env python

from AngleCalculator import generateAngles
from C45Tree import C45Tree
from RandomForest import RandomForest
from Data import TrainingData, sample
import sys
import numpy

#An array of all the files containing data and an array of the labels for each file 
dataFiles = [ "./data/PositionsDisco.log", "./data/PositionsChkn.log", "./data/PositionsEgy.log", "./data/Positionsymca.log"]
labels = ["Disco", "ChickenDance", "WalkLikeAnEgyptian", "YMCA"]

def generateData():
    trainingData = TrainingData("Testing Data")
    index = 0

    for filename in dataFiles:
        #Open the file
        fIn = open(filename,'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()
        index += 1

    return trainingData




def main():
    theData = generateData()
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testFile = open("./data/testData.log", 'r')
    sampleList = []
    labels = ["YMCA", "Disco", "WalkLikeAnEgyptian", "ChickenDance"]
    index = 0

    for line in testFile:
        features = generateAngles(line)
        sampleList.append(sample(features, labels[index]))
        index += 1

    for samp in sampleList:
        print samp.getLabel()
        print testForest.classify(samp)

if __name__ == '__main__':
    main()

