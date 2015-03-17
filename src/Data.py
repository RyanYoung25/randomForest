#! usr/bin/env python
import numpy as np
import random
#Author: Ryan Young


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

    def normalizeValues(self, means, stdDevs):
        '''
        Perform Z score normalization on the values 
        for this sample
        '''
        index = 0
        for index in xrange(len(self.features)):
            z = (self.features[index] - means[index]) / stdDevs[index]
            self.features[index] = z
            index += 1

    def setLabel(self, label):
        #Set the label for the data sample
        self.label = label

    def getFeatures(self):
        #Get the feature vector
        return self.features

    def getValueAtIndex(self, index):
        #Get the value at the index
        return self.features[index]

    def getLabel(self):
        #Get the label for this sample
        return self.label

    def getRandomSubsetFeatures(self, seed=0, num=0):
        #Set the seed if necessary
        if seed != 0:
            random.seed(seed)
        if num == 0:
            #Default sqrt(p)
            num = np.sqrt(len(self.features))

        subFeatures = random.sample(self.features, num)
        subFeatures.shape = (len(subFeatures),)  #Change the shape back to a single dimensional numpy array
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
    def __init__(self, DataName, data=None):
        self.DataName = DataName
        if data is None:
            self.data = []
        else:
            self.data = data  #Data is a simple list of samples
        self.stats = None
        self.entropy = None

    def addSample(self, sample):
        #Add a sample to the data set
        if sample.getLabel() is not None:
            self.data.append(sample)

    def getData(self):
        #Return the data list
        return self.data

    def getLength(self):
        #Return the length of the data list
        return len(self.data)

    def getFeatureLength(self):
        #Return the length of the features in this data
        # this assumes that the data list is populated and
        # uniform in sample type.
        return len(self.data[0].getFeatures())

    def isPureA(self):
        '''
        Checks to see if all of the data samples in this set 
        are from the same class. This is a version that took 
        longer than the other version
        '''

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
        if firstType is None:
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

        if self.stats is not None:
            return self.stats

        stats = {}
        for elem in self.data:
            label = elem.getLabel()
            if label in stats:
                stats[label] += 1
            else:
                stats[label] = 1

        self.stats = stats

        return self.stats

    def getEntropy(self):
        '''
        Returns a numerical quantity of the entropy for this 
        data set. 
        '''

        if self.entropy is not None:
            return self.entropy

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

        self.entropy = entropy
        return self.entropy

    #Might take a while this way
    def getBestThreshold(self, feature):
        #Calculate which feature value splits 
        #the best, has the lowest entropy
        
        minEnt = float("inf")
        bestThreshold = 0
        for samp in self.data:
            thresh = samp.getFeatures()[feature] 
            (LSet, RSet) = self.splitOn(feature, thresh)
            LEnt = LSet.getEntropy()
            REnt = RSet.getEntropy()
            newEnt = LEnt + REnt
            if newEnt <= minEnt:
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
        if seed != 0:
            random.seed(seed)

        bag = []

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
        #Partition the dataset into k equal segements. This 
        # is used to create the sets for crossvalidation

        randomDatalist = list(self.data)
        random.shuffle(randomDatalist)
        sliceSize = len(randomDatalist) / k

        listOfData = []

        for i in range(0, len(randomDatalist), sliceSize):
            slice = randomDatalist[i:i+sliceSize]
            listOfData.append(TrainingData("slice", slice))

        return listOfData

    def normalizeData(self):
        '''
        Calculate the mean and standard deviation for every 
        feature value then normalize all of the data. This is 
        Z score normalization and is important for the position 
        data. 

        This returns a tuple of lists that contain the mean 
        and standard deviation for each feature value so that it can
        be applied to future data samples
        '''

        n = len(self.data)
        sums = [0 for i in xrange(len(self.data[0].getFeatures()))]
        squaredSums = [0 for i in xrange(len(self.data[0].getFeatures()))]
        means = [0 for i in xrange(len(self.data[0].getFeatures()))]
        stdDevs = [0 for i in xrange(len(self.data[0].getFeatures()))]

        for samp in self.data:
            features = samp.getFeatures()
            index = 0
            for val in features:

                sums[index] += val
                squaredSums[index] += val * val
                index += 1

        for i in xrange(len(sums)):
            means[i] = sums[i] / n
            stdDevs[i] = np.sqrt((squaredSums[i]/n) - (means[i] * means[i]))

        for samp in self.data:
            samp.normalizeValues(means, stdDevs)

        return (means, stdDevs)

    def combineWithNewData(self, newData):
        #Combine this data set with another one
        self.data += newData.getData()

