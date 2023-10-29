import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

class SO3(object):
    """
    Class to capture the orientation component of SE3
    """
    @staticmethod
    def I():
        """
        Return 3x3 Identity Matrix
        """
        return np.diag(np.array([1, 1, 1]))
    
    @staticmethod
    def multiplication(leftMat, rightMat):
        output =np.matmul(leftMat, rightMat)

        return output
    
    @staticmethod
    def rotationAboutZ(angle):
        """
        Return a rotation about Z axis
        """
        s = np.sin(angle)
        c = np.cos(angle)
        return np.array([[c, -s, 0], 
                         [s, c, 0], 
                         [0, 0, 1]])
    
    @staticmethod
    def rotationAboutY(angle):
        """
        Rotation about Y axis
        """
        s = np.sin(angle)
        c = np.cos(angle)
        return np.array([[c, 0, -s], 
                         [0, 1, 0], 
                         [s, 0, c]])

    @staticmethod
    def rotationAboutX(angle):
        """
        Rotation about X axis
        """
        s = np.sin(angle)
        c = np.cos(angle)
        return np.array([[1, 0, 0], 
                         [0, c, -s], 
                         [0, s, c]])

class SE3:
    """
    Class to capture SE3 Transformation Matrices
    """   
    @staticmethod
    def toSE3(R, r):
        T_nh = np.concatenate((R,r), axis=1)
        T_h = np.concatenate((T_nh,np.array([[0, 0, 0, 1]])), axis=0)
        
        return T_h

    @staticmethod
    def multiplication(leftMat, rightMat):
        output =np.matmul(leftMat, rightMat)

        return output

    @staticmethod
    def setPosition(T, newPos):
        pass

    @staticmethod
    def getPosition(T):
        return T[0:3,3]
    
    @staticmethod
    def setRotation(T, newRot):
        pass
    
    @staticmethod
    def getRotation(T):
        return T[0:3, 0:3]
        
class Output:
    """
    Class used to write the youBot state to a csv
    """    
    def __init__(self, csvFileName, fieldNames):  
        self.csvTable = []
        self.csvFileName = csvFileName
        self.fieldNames = fieldNames

    def AddState(self, state):
        stateDict = {}
        idx = 0
        for name in self.fieldNames:
            stateDict[name] = state[idx]
            idx += 1
        self.csvTable.append(stateDict)

    def WriteStateToCSV(self):
        with open(self.csvFileName, 'w') as csvfile:
            csvWriter = csv.DictWriter(csvfile, fieldnames=self.fieldNames)
            for entry in self.csvTable:
                csvWriter.writerow(entry)
    
    def plotState(self):
        data_df = pd.DataFrame(self.csvTable)
        plot = data_df.plot()
        plt.show()
        
    
