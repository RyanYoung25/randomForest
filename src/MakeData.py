#!/usr/bin/env python

import json
import numpy as np
import sys

JOINTS = [
        'head',
        'neck',
        'torso',
        'left_shoulder',
        'left_elbow',
        'left_hand',
        'left_hip',
        'left_knee',
        'left_foot',
        'right_shoulder',
        'right_elbow',
        'right_hand',
        'right_hip',
        'right_knee',
        'right_foot',
        ]

def returnLine(jsonLine):
    #
    dict = json.loads(jsonLine)
    #Initialize the empty line
    line = []
    jointList = dict["Joints"]

    #For each joint in the joint array, get the pos x, y, z, and rot x, y, z, w
    for joint in jointList:
        positions = joint["pos"]
        line.append(positions["x"])
        line.append(positions["y"])
        line.append(positions["z"])  

    newline = np.array(line)
    return newline


def main():
    filename = "Positions.log"
    output = "Joints.csv"

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) == 3:
        output = sys.argv[2]
    elif len(sys.argv) > 3:
        print "Usage: ./jsonToMatrix.py <Input File> <Output File>"

    #Open the input file
    fIn = open(filename,'r')
    #Open the output file  
    fOut = open(output, 'w')  
    #Iterate through line by line of the input 
    #creating a row vector for each line in the output
    for line in fIn:
        vector = returnLine(line)
        fOut.write(str(vector) + "\n")
        

if __name__ == '__main__':
    main()