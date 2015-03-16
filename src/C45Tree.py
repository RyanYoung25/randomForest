#!/usr/bin/env python
from Data import sample, TrainingData
import random
import numpy as np
import sys


class TreeNode:
    """docstring for treeNode"""
    def __init__(self, dataSet, featureList, parent=None):
        self.featureNumber = None #This is the trained index of the feature to split on
        self.featureList = featureList 
        self.threshold = None     #This is the trained threshold of the feature to split on
        self.leftChild = None
        self.rightChild = None
        self.dataSet = dataSet
        self.parent = parent

    def c45Train(self):
        '''
        The feature list is an array of the possible 
        feature indicies to use. This prevents splitting on 
        the same feature multiple times

        Runs a rough C45 algorithm. 

        In pseudocode, the general algorithm for building decision trees is:

        1. Check for base cases
        2. For each attribute a
            3. Find the normalized information gain ratio from splitting on a
        4. Let a_best be the attribute with the highest normalized information gain
        5. Create a decision node that splits on a_best
        6. Recur on the sublists obtained by splitting on a_best, and add those nodes as children of node
        '''

        #Base Cases:
        
        #All instances in dataSet are the same
        if(self.dataSet.isPure()):
            #gets the label of the first data instance and makes a leaf node
            #classifying it. 
            label =  self.dataSet.getData()[0].getLabel()
            leaf = LeafNode(label)
            return leaf
        #If there are no more features in the feature list
        if len(self.featureList) == 0:
            labels = self.dataSet.getLabelStatistics()
            bestLabel = None
            mostTimes = 0

            for key in labels:
                if labels[key] > mostTimes:
                    bestLabel = key
                    mostTimes = labels[key]
            #Make the leaf node with the best label
            leaf = LeafNode(bestLabel)
            return leaf

        
        
        #Check all of the features for the split with the most 
        #information gain. Use that split.
        currentEntropy = self.dataSet.getEntropy()
        currentLength = self.dataSet.getLength()
        infoGain = -1 * sys.maxint
        bestFeature = 0
        bestLeft = None
        bestRight = None
        bestThreshold = 0

        #Feature Bagging, Random subspace
        num = int(np.ceil(np.sqrt(len(self.featureList))))
        featureSubset = random.sample(self.featureList, num)

        for featureIndex in featureSubset:
            #Calculate the threshold to use for that feature
            threshold = self.dataSet.betterThreshold(featureIndex)

            (leftSet, rightSet) = self.dataSet.splitOn(featureIndex, threshold)

            leftEntropy = leftSet.getEntropy()
            rightEntropy = rightSet.getEntropy()
            #Weighted entropy for this split
            newEntropy = (leftSet.getLength() / currentLength) * leftEntropy + (rightSet.getLength() / currentLength) * rightEntropy
            #Calculate the gain for this test
            newIG = currentEntropy - newEntropy

            if(newIG > infoGain):
                #Update the best stuff
                infoGain = newIG
                bestLeft = leftSet
                bestRight = rightSet
                bestFeature = featureIndex
                bestThreshold = threshold

        newFeatureList = list(self.featureList)
        newFeatureList.remove(bestFeature)

        #Another base case, if there are no good features to split on
        if bestLeft.getLength() == 0 or bestRight.getLength() == 0:
            labels = self.dataSet.getLabelStatistics()
            bestLabel = None
            mostTimes = 0

            for key in labels:
                if labels[key] > mostTimes:
                    bestLabel = key
                    mostTimes = labels[key]
            #Make the leaf node with the best label
            leaf = LeafNode(bestLabel)
            return leaf

        self.threshold = bestThreshold
        self.featureNumber = bestFeature


        leftChild = TreeNode(bestLeft, newFeatureList, self)
        rightChild = TreeNode(bestRight, newFeatureList, self)

        self.leftChild = leftChild.c45Train()
        self.rightChild = rightChild.c45Train()

        return self
        
    def __str__(self):
        return str(self.featureList)

    def __repr__(self):
        return self.__str__()
                
    def classify(self, sample):
        '''
        Recursivly traverse the tree to classify the sample that is passed in. 
        '''


        value = sample.getFeatures()[self.featureNumber]

        if(value < self.threshold):
            #Continue down the left child    
            return self.leftChild.classify(sample)

        else:
            #continue down the right child
            return self.rightChild.classify(sample)

class LeafNode:

    def __init__(self, classification):
        self.classification = classification

    def classify(self, sample):
        #A leaf node simply is a classification, return that
        #This is the base case of the classify recursive function for TreeNodes
        return self.classification

class C45Tree:

    def __init__(self, data):
        self.rootNode = None
        self.data = data

    def train(self):
        '''
        Trains a decision tree classifier on data set passed in. 
        The data set should contain a good mix of each class to be
        classified.
        '''
        length  = self.data.getFeatureLength()
        featureIndices = range(length)
        self.rootNode = TreeNode(self.data, featureIndices)
        self.rootNode.c45Train()

    def classify(self, sample):
        '''
        Classify a sample based off of this trained tree.
        '''

        return self.rootNode.classify(sample)


