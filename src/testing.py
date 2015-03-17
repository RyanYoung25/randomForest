#!/usr/bin/env python

from AngleCalculator import generateAngles
from RandomForest import RandomForest
from Data import TrainingData, sample
from MakeData import returnLine
import copy
#Author: Ryan Young

#An array of all the files containing data and an array of the labels for each file 
labels = ["Disco", "ChickenDance", "WalkLikeAnEgyptian", "YMCA"]
sub1 = ["./data/PositionsDisco.log", "./data/PositionsChkn.log", "./data/PositionsEgy.log", "./data/Positionsymca.log"]
sub2 = ["./data/TestDisco.log", "./data/TestChkn.log", "./data/TestEgy.log", "./data/TestYMCA.log"]
sub3 = ["./data/JDisco.log", "./data/JChkn.log", "./data/JEgy.log", "./data/JYMCA.log"]


class confusionMatrix:
    '''
    This is just a helper class for creating a confusion matrix
    It contains a map of each label where the first key is the true 
    label and each value is another map where the second key is the
    predicted label. From there we can increment a value for that 
    prediction. 
    '''

    def __init__(self, labelList):

        self.matrix = {}
        for label in labelList:
            self.matrix[label] = dict((key, 0) for key in labelList)

    def update(self, actual, predicted):
        self.matrix[actual][predicted] += 1

    def printMatrix(self):
        for label in self.matrix.keys():
            for nextLabel in self.matrix[label].keys():
                print self.matrix[label][nextLabel],
            print '\n'


def generateAllAngleTrainingData():
    '''
    This function creates one large TrainingData object 
    containing data from all three participants but uses
    calculated angles as the features for each data sample.
    '''
    trainingData = TrainingData("Testing Data")
    index = 0

    for i in xrange(len(labels)):
        #Open the file
        fIn = open(sub1[i], 'r')
        f2In = open(sub2[i], 'r')
        f3In = open(sub3[i], 'r')

        #For each line of the files calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()

        for line in f2In:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f2In.close()

        for line in f3In:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f3In.close()

        index += 1

    #Return the data object
    return trainingData


def generateAllPositionTrainingData():
    '''
    This function creates one large TrainingData object 
    containing data from all three participants but uses
    regular position data as the features for each data sample.
    '''
    trainingData = TrainingData("Testing Data")
    index = 0

    for i in xrange(len(labels)):
        #Open the file
        fIn = open(sub1[i], 'r')
        f2In = open(sub2[i], 'r')
        f3In = open(sub3[i], 'r')

        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the trainingData.
        for line in fIn:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()

        for line in f2In:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f2In.close()

        for line in f3In:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f3In.close()

        index += 1

    #Return the data
    return trainingData   


def generateTwoAngleTrainingData():
    '''
    This function creates training data using the first
    two participants' data and has joint angles as features
    for the samples.
    '''

    trainingData = TrainingData("Testing Data")
    index = 0

    for i in xrange(len(labels)):
        #Open the file
        fIn = open(sub1[i], 'r')
        f2In = open(sub2[i], 'r')

        #For each line of the files calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()

        for line in f2In:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f2In.close()

        index += 1

    return trainingData


def generateTwoPositionTrainingData():
    '''
    This function creates training data using the first
    two participants' data and has joint positions as features
    for the samples.
    '''

    trainingData = TrainingData("Testing Data")
    index = 0

    for i in xrange(len(labels)):
        #Open the file
        fIn = open(sub1[i], 'r')
        f2In = open(sub2[i], 'r')

        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the trainingData.
        for line in fIn:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()

        for line in f2In:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        f2In.close()

        index += 1

    return trainingData


def generateOneTestAngleData():
    '''
    This creates a list of sample objects that can 
    be used for testing. This list is built from the third 
    participant's data and uses angles as features.
    '''

    testSamples = []
    index = 0

    for filename in sub3:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the list.
        for line in fIn:
            features = generateAngles(line)
            testSamples.append(sample(features, labels[index]))
        fIn.close()
        index += 1

    return testSamples


def generateOneTestPositionData(means, stdDevs):
    '''
    This creates a list of sample objects that can 
    be used for testing. This list is built from the third 
    participant's data and uses joint positions as features.
    The poisitions are also normalized as they are added to 
    the list.
    '''

    testSamples = []
    index = 0

    for filename in sub3:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the list.
        for line in fIn:
            features = returnLine(line)
            samp = sample(features, labels[index])
            samp.normalizeValues(means, stdDevs)
            testSamples.append(samp)
        fIn.close()
        index += 1

    return testSamples


def generateTwoTestAngleData():
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the calculated angles are used as features.
    '''
    testSamples = []
    index = 0

    for filename in sub2:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the list.
        for line in fIn:
            features = generateAngles(line)
            testSamples.append(sample(features, labels[index]))
        fIn.close()
        index += 1

    index = 0
    for filename in sub3:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the list.
        for line in fIn:
            features = generateAngles(line)
            testSamples.append(sample(features, labels[index]))
        fIn.close()
        index += 1

    return testSamples


def generateTwoTestPositionData(means, stdDevs):
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the joint poisitions are used as features.
    '''
    testSamples = []
    index = 0

    for filename in sub2:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the list.
        for line in fIn:
            features = returnLine(line)
            samp = sample(features, labels[index])
            samp.normalizeValues(means, stdDevs)
            testSamples.append(samp)
        fIn.close()
        index += 1

    index = 0 

    for filename in sub3:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the list.
        for line in fIn:
            features = returnLine(line)
            samp = sample(features, labels[index])
            samp.normalizeValues(means, stdDevs)
            testSamples.append(samp)
        fIn.close()
        index += 1

    return testSamples


def generateOneTrainAngleData():
    '''
    This creates a data set to be used to train a classifier.
    The data is from participant 1 and uses the calculated
    joint angles as features.
    '''
    trainingData = TrainingData("Angle Data")
    index = 0

    for filename in sub1:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the file calculate the 
        # angles inbetween joints and use the resulting 
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = generateAngles(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()
        index += 1

    return trainingData


def generateOneTrainPositionData():
    '''
    This creates a data set to be used to train a classifier.
    The data is from participant 1 and uses the joint positions
    as features.
    '''
    trainingData = TrainingData("Position Data")
    index = 0

    for filename in sub1:
        #Open the file
        fIn = open(filename, 'r')
        #For each line of the files, create a feature vector
        # of the raw joint positions. Add that to the trainingData.
        for line in fIn:
            features = returnLine(line)
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()
        index += 1

    return trainingData


def crossValidationAngles():
    '''
    Performs 10 fold cross validation on the total angle 
    dataset
    '''
    theData = generateAllAngleTrainingData() 
    k = 10

    #Partition the data into 10 subsets
    dataSets = theData.getKSegments(k)

    #For each of the 10 subsets leave one out, train on the 
    # other 9, test on the one left out, print the accuracy. 
    results = confusionMatrix(labels)
    for i in xrange(k):
        print i
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
        
        for samp in testSet.getData():
            resultLabel = testForest.classify(samp)
            trueLabel = samp.getLabel()
            results.update(trueLabel, resultLabel)

    results.printMatrix()


def crossValidationPositions():
    '''
    Performs 10 fold cross validation on the total 
    joint position dataset
    '''
    theData = generateAllPositionTrainingData() 
    means, stdDevs = theData.normalizeData()
    k = 10

    #Partition the data into 10 subsets
    dataSets = theData.getKSegments(k)

    #For each of the 10 subsets leave one out, train on the 
    # other 9, test on the one left out, print the accuracy. 
    results = confusionMatrix(labels)
    for i in xrange(k):
        print i
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
        
        for samp in testSet.getData():
            resultLabel = testForest.classify(samp)
            trueLabel = samp.getLabel()

            results.update(trueLabel, resultLabel)

    results.printMatrix()


def oneVsTwoAngles():
    '''
    Trains a random forest on the data from participant 1 
    and tests it on participant 2 and 3. The data used here 
    uses the angle features
    '''
    theData = generateOneTrainAngleData()
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testList = generateTwoTestAngleData()

    results = confusionMatrix(labels)

    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        results.update(trueLabel, resultLabel)

    results.printMatrix()


def oneVsTwoPositions():
    '''
    Trains a random forest on the data from participant 1 
    and tests it on participant 2 and 3. The data used here 
    uses the position features
    '''
    theData = generateOneTrainPositionData()
    means, stdDevs = theData.normalizeData()
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testList = generateTwoTestPositionData(means, stdDevs)

    results = confusionMatrix(labels)

    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        results.update(trueLabel, resultLabel)

    results.printMatrix()


def twoVsOneAngles():
    '''
    Trains a random forest on the data from participants 1 and 2 
    and tests it on participant 3. The data used here 
    uses the angle features
    '''
    theData = generateTwoAngleTrainingData()
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testList = generateOneTestAngleData()

    results = confusionMatrix(labels)

    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        results.update(trueLabel, resultLabel)

    results.printMatrix()


def twoVsOnePositions():
    '''
    Trains a random forest on the data from participants 1 and 2 
    and tests it on participant 3. The data used here 
    uses the angle features
    '''
    theData = generateTwoPositionTrainingData()
    means, stdDevs = theData.normalizeData()
    testForest = RandomForest(theData)
    print "Training"
    testForest.train()
    print "Done!"

    testList = generateOneTestPositionData(means, stdDevs)

    results = confusionMatrix(labels)

    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        results.update(trueLabel, resultLabel)

    results.printMatrix()


if __name__ == '__main__':
    print "1v2 Angles: "
    oneVsTwoAngles()
    print "1v2 Positions: "
    oneVsTwoPositions()
    print "2v1 Angles: "
    twoVsOneAngles()
    print "2v1 Positions: "
    twoVsOnePositions()
    print "Angles CrossVal: "
    crossValidationAngles()
    print "Positions CrossVal: "
    crossValidationPositions()

