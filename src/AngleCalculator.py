#! /usr/bin/env python

import numpy as np
import math 
import json
import sys

'''
Calculate the theta between two vectors a, and b. 
'''
def calculateTheta(a, b):
    dotProd = np.dot(a, b)        #a dot b
    aNorm = np.sqrt(np.dot(a, a)) # ||a||
    bNorm = np.sqrt(np.dot(b, b)) # ||b||
    #theta = arccos(a dot b / (||a||*||b||))
    return math.acos(dotProd/(aNorm*bNorm))

'''
Takes a list of xyz positions for two points, A and
B and generates the vector from A to B
'''
def generateVector(Apos, Bpos):
    #B minus A
    xComp = Bpos[0] - Apos[0]
    yComp = Bpos[1] - Apos[1]
    zComp = Bpos[2] - Apos[2]
    return np.array([xComp, yComp, zComp])
'''
Return a numpy vector of the coordinates for 
the joint struture passed to to
'''
def getPointFromJoint(joint):
    posList = []
    posList.append(joint['pos']['x'])
    posList.append(joint['pos']['y'])
    posList.append(joint['pos']['z'])
    return np.array(posList)


'''
Math for calculating the Shoulder Yaw from
Matthew Wiese Star Project ITERO
'''
def calculateYaw(EW, ES, joint):
        #finding z line cross shoulder elbow vector:
        avec1 = np.array([joint[0], joint[1], 2])
        bvec1 = ES 
        zcross = np.cross(avec1, bvec1)
        EO = np.cross(EW, ES)
        EB = joint
        #finding cross of zcrossR and bvec
        zCross2 = np.cross(bvec1, zcross) 
        #LEO = (LEO / norm(REO)) + EB
        step1EO = EO / np.linalg.norm(EO)
        EO = step1EO + EB
        #zCross2 = (zCross2 / norm(zCross2)) + LE
        step1zCross2 = zCross2 / np.linalg.norm(zCross2)
        zCross2 = step1zCross2 + EB

        elbowToOrentation = generateVector(EB, EO)
        elbowToZ = generateVector(EB, zCross2)

        theta = calculateTheta(elbowToOrentation, elbowToZ)
        return theta

def generateAngles(jsonString):
    #Make the jsonString into a dictionary to get the data
    dict = json.loads(jsonString)
    '''
    Order for joint angles: [REB, LEB, RSY, LSY, RSR, LSR, RSP, LSP]
    To calculate each angle, take the angle between the vectors formed
    by the specified points*:

    REB- RE-RW, RE-RS
    LEB- LE-LW, LE-LS
    RSY*-
    LSY*-
    RSR- RS-NK, RS-RE
    LSR- LS-NK, LS-LE
    RSP**- cross(RSR), NK-HD
    LSP**- cross(LSR), NK-HD

    *The shoulder yaws require a bit more complexity. First we must 
    make a constant vector in the z direction and take the cross product of the two vectors
    specified. Then we need to cross that vector with the constant vector and get the angle between 
    that vector and the vector from the cross product of the orginal vectors

    JOINTS = [
       1 'head',
       2 'neck',
       3 'torso',
       4 'left_shoulder',
       5 'left_elbow',
       6 'left_hand',
       7 'left_hip',
       8 'left_knee',
       9 'left_foot',
       10 'right_shoulder',
       11 'right_elbow',
       12 'right_hand',
       13 'right_hip',
       14 'right_knee',
       15 'right_foot',
        ]
    '''
    #Get the joint list
    jointList = dict["Joints"]

    #Create coordianate vectors for each joint point
    #Indices from above
    RW = getPointFromJoint(jointList[12])
    RE = getPointFromJoint(jointList[11])
    RS = getPointFromJoint(jointList[10])
    NK = getPointFromJoint(jointList[2])
    HD = getPointFromJoint(jointList[1])
    LS = getPointFromJoint(jointList[4])
    LE = getPointFromJoint(jointList[5])
    LW = getPointFromJoint(jointList[6])

    #Generate the vectors that we will need to calculate the angles
    REBA = generateVector(RE, RW)
    REBB = generateVector(RE, RS)
    LEBA = generateVector(LE, LW)
    LEBB = generateVector(LE, LS)

    RSYA = generateVector(RE, RW)
    RSYB = generateVector(RE, RS)
    LSYA = generateVector(LE, LW)
    LSYB = generateVector(LE, LS)

    RSRA = generateVector(RS, NK)
    RSRB = generateVector(RS, RE)
    LSRA = generateVector(LS, NK)
    LSRB = generateVector(LS, LE)
    RSPA = np.cross(RSRA, RSRB)
    RSPB = generateVector(NK, HD)
    LSPA = np.cross(LSRA, LSRB)
    LSPB = RSPB

    #Generate the angles
    REB = calculateTheta(REBA, REBB)
    LEB = calculateTheta(LEBA, LEBB)
    RSY = calculateYaw(RSYA, RSYB, RE)
    LSY = calculateYaw(LSYA, LSYB, LE)
    RSR = calculateTheta(RSRA, RSRB)
    LSR = calculateTheta(LSRA, LSRB)
    RSP = calculateTheta(RSPA, RSPB)
    LSP = calculateTheta(LSPA, LSPB)

    return np.array([REB, LEB, RSY, LSY, RSR, LSR, RSP, LSP])


if __name__ == '__main__':
    filename = "Positions.log"
    output = "JointAngles.txt"

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) == 3:
        output = sys.argv[2]
    elif len(sys.argv) > 3:
        print "Usage: ./AngleCalculator.py <Input File> <Output File>"

    #Open the input file
    fIn = open(filename,'r')
    #Open the output file  
    fOut = open(output, 'w')  
    #Iterate through line by line of the input 
    #creating a row vector for each line in the output
    for line in fIn:
        vector = generateAngles(line)
        fOut.write(str(vector) + "\n")