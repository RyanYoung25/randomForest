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
        the threshold for that attribute number. If the attribute 
        number does not exist it returns false
        '''

        if len(self.features) >= attributeNumber:
            return False

        return self.features[attributeNumber] <= threshold



class TrainingData:
    """A TrainingData object contains a collection of samples that have labels.
    This can be passed to a classifier and be used for training. """
    def __init__(self, DataName):
        self.DataName = DataName
        self.data = []  #Data is a simple list of samples

    def addSample(self, sample):
        if sample.getLabel() != None:
            self.data.append(sample)

    def getData(self):
        return self.data

    def getLength(self):
        return len(self.data)

    def isPure(self):
        '''
        Returns true if all of the data elements have the same 
        class type and that class type is not None.
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
        if firstType == None:
            return False

        #On success
        return True


    def splitOn(self, attributeNumber, threshold):
        '''
        Splits this dataset instance into two subsets 
        based off of the threshold for the attributeNumber. 

        Returns a 2 tuple the less-than-or-equal-to set 
        and the greater-than set
        '''

        leftData = []
        rightData = []

        for elem in self.data:
            if elem.splitLeft(attributeNumber, threshold):
                leftData.append(elem)
            else:
                rightData.append(elem)

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
        entropy = 0 

        for key in classes:
            #Calculate the probability of the key
            pOfKey = classes[key] / total
            #Calculate the entropy for this key and add it to the running sum
            entropy = entropy + -1 * pOfKey * np.log2(pOfKey)

        return entropy


    def getRandomSubset(self, seed=0, num=0):
        #Set the seed if necessary
        if seed !=0:
            random.seed(seed)
        if num == 0:
            #Default length of half the data set
            num = len(self.data) / 2

        #Choose the number of elements from the subset
        random.sample(self.data, num)    