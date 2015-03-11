#!/usr/bin/env python

from AngleCalculator import generateAngles
from C45Tree import C45Tree
from RandomForest import RandomForest
from Data import TrainingData, sample
import sys
import copy
import numpy

#An array of all the files containing data and an array of the labels for each file 
dataFiles = [ "./data/PositionsDisco.log", "./data/PositionsChkn.log", "./data/PositionsEgy.log", "./data/Positionsymca.log"]
labels = ["Disco", "ChickenDance", "WalkLikeAnEgyptian", "YMCA"]
newDataFiles = ["./data/TestDisco.log", "./data/TestChkn.log", "./data/TestEgy.log", "./data/TestYMCA.log"]

def generateTrainingData():
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


def generateTestData():
    testSamples = []
    index = 0

    for filename in newDataFiles:
        #Open the file
        fIn = open(filename,'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = generateAngles(line)
            testSamples.append(sample(features, labels[index]))
        fIn.close()
        index += 1

    return testSamples


def stdTest():
    theData = generateTrainingData()   
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

def crossValidation():
    theData = generateTrainingData() 
    k= 10

    #Partition the data into 10 subsets
    dataSets = theData.getKSegments(k)

    #For each of the 10 subsets leave one out, train on the 
    # other 9, test on the one left out, print the accuracy. 
    results = {"Correct":0, "Incorrect":0}
    for i in xrange(k):
        #testing set
        testSet = dataSets[i]
        #Build the training set
        trainingSet = TrainingData("CrossVal")
        trainingList = copy.deepcopy(dataSets)
        trainingList.pop(i)
        for elem in trainingList:
            trainingSet.combineWithNewData(elem)

        #train the classifier on the trainingSet
        testForest = RandomForest(trainingSet)
        testForest.train()

        #Evaluate the classifer on the test set
        
        for sample in testSet.getData():
            resultLabel = testForest.classify(sample)
            trueLabel = sample.getLabel()

            if resultLabel == trueLabel:
                results["Correct"] += 1
            else:
                results["Incorrect"] += 1

    print results

def testOnNewData():
    theData = generateTrainingData()   
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testList = generateTestData()

    results = {"Correct":0, "Incorrect":0}

    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        if resultLabel == trueLabel:
            results["Correct"] += 1
        else:
            results["Incorrect"] += 1

    print results

if __name__ == '__main__':
    testOnNewData()

