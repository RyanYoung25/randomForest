#! usr/bin/env python
import numpy as np
import random

class sample:
    """A sample object is one instance of a sample. It contains a feature
    vector and may or may not contain a classification label. If the sample is 
    supposed to be used for supervised training it requires a label. If a sample
    is to be classified the label is not required  """
    def __init__(self, features=None, label=None):
        self.features = features
        self.label = label

    def setFeatures(self, features):
        #Check if the features are a single dimensional numpy array 
        if (type(features) != np.ndarray) or (len(features.shape) != 1):
            raise ValueError("The feature vector is not a single dimensional numpy array!")

        self.features = features

    def setLabel(self, label):
        #Set the label for the data sample
        self.label = label

    def getFeatures(self):
        #Get the feature vector
        return self.features

    def getValueAtIndex(self, index):
        return self.features[index]

    def getLabel(self):
        #Get the label for this sample
        return self.label

    def getRandomSubsetFeatures(self, seed=0, num=0):
        #Set the seed if necessary
        if seed !=0:
            random.seed(seed)
        if num == 0:
            #Default sqrt(p)
            num = np.sqrt(len(self.features))

        subFeatures = random.sample(self.features, num)
        subFeatures.shape = (len(subFeatures),) #Change the shape back to a single dimensional numpy array
        return subFeatures

    def splitLeft(self, attributeNumber, threshold):
        '''
        Returns true if the sample is less-than-or-equal-to 
        the threshold for that attribute number. 
        '''

        if self.features[attributeNumber] <= threshold:
            return True

        return False



class TrainingData:
    """A TrainingData object contains a collection of samples that have labels.
    This can be passed to a classifier and be used for training. """
    def __init__(self, DataName, data=[]):
        self.DataName = DataName
        self.data = data #Data is a simple list of samples

    def addSample(self, sample):
        if sample.getLabel() != None:
            self.data.append(sample)

    def getData(self):
        return self.data

    def getLength(self):
        return len(self.data)

    def getFeatureLength(self):
        return len(self.data[0].getFeatures())

    def isPureA(self):

        #If there is no data 
        if len(self.data) <= 0:
            return False


        firstType = self.data[0].getLabel()
        #For each sample check if the label matches the first
        # if they are all the same they all will
        for elem in self.data:
            if(elem.getLabel() != firstType):
                return False

        #If the type that they all matched was None the data 
        # was unlabled so we can't call it pure
        if firstType == None:
            return False

        #On success
        return True

    def isPure(self):
        '''
        Returns true if all of the data elements have the same 
        class type
        '''

        labels = self.getLabelStatistics()

        if len(labels.keys()) != 1:
            return False

        return True


    def splitOn(self, attributeNumber, threshold):
        '''
        Splits this dataset instance into two subsets 
        based off of the threshold for the attributeNumber. 

        Returns a 2 tuple the less-than-or-equal-to set 
        and the greater-than set
        '''

        left = []
        right = []

        for elem in self.data:
            if elem.splitLeft(attributeNumber, threshold):
                left.append(elem)
            else:
                right.append(elem)

        leftData = TrainingData("left", left)
        rightData = TrainingData("right", right)


        return (leftData, rightData)


    def getLabelStatistics(self):
        '''
        Returns a dictionary of each label for all the data and 
        the number of occurrences for that label
        '''

        stats = {}

        for elem in self.data:
            label = elem.getLabel()
            if label in stats:
                stats[label] += 1
            else:
                stats[label] = 1

        return stats


    def getEntropy(self):
        '''
        Returns a numerical quantity of the entropy for this 
        data set. 
        '''

        classes = self.getLabelStatistics()

        #assuming all the data is labeled 
        total = len(self.data)
        entropy = 0.0 

        for key in classes:
            #Calculate the probability of the key
            pOfKey = float(classes[key]) / float(total)
            #Calculate the entropy for this key and add it to the running sum
            if pOfKey != 0:
                entropy = entropy + -1 * pOfKey * np.log2(pOfKey)


        return entropy

    #Might take a while this way
    def getBestThreshold(self, feature):
        #Calculate which feature value splits 
        #the best, has the lowest entropy
        minEnt = 1
        bestThreshold = 0
        for samp in self.data:
            thresh = samp.getFeatures()[feature] 
            (LSet, RSet) = self.splitOn(feature, thresh)
            LEnt = LSet.getEntropy()
            REnt = RSet.getEntropy()
            newEnt = LEnt + REnt
            if newEnt < minEnt:
                minEnt = newEnt
                bestThreshold = thresh

        return bestThreshold


    def betterThreshold(self, feature):
        #Calculate the average value, split on that. 
        totalN = self.getLength()
        runningTotal = 0.0

        for samp in self.data:
            runningTotal += samp.getValueAtIndex(feature)

        return float(runningTotal) / totalN



    def getBag(self, seed=0):
        #Set the seed if necessary
        if seed !=0:
            random.seed(seed)

        bag =[]

        #Create the bag
        for i in range(0, len(self.data)):
            bag.append(random.choice(self.data))

        #TrainingData bag 
        bagSet = TrainingData("bag", bag)
        return bagSet


    def addSampleFromFeatures(self, features, label):
        #Make a sample object and add it to the data list
        samp = sample(features, label)
        self.addSample(samp)

    def getKSegments(self, k):

        randomDatalist = list(self.data)
        random.shuffle(randomDatalist)
        sliceSize = len(randomDatalist) / k

        listOfData = []

        for i in range(0, len(randomDatalist), sliceSize):
            slice = randomDatalist[i:i+sliceSize]
            listOfData.append(TrainingData("slice", slice))

        return listOfData

    def combineWithNewData(self, newData):
        self.data += newData.getData()

