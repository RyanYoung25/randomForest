#! usr/bin/env python

import numpy as np 


class RandomForest(object):
    """A random forest object with the default of using a C45Tree 
    for each of the trees in the random forest. To train the forest
    create an instance of it then call train on a TraingData object"""
    def __init__(self, arg):
        #TODO
        