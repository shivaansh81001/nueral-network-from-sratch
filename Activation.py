import numpy as np

class Activation:
    def __init__(self, inputs):
        self.Z = inputs
    
    def ReLU(self):
        '''returns ReLU activated results'''
        return np.maximum(0,self.Z)
    
    def sigmoid(self):
        '''returns sigmoid activated results'''
        return 1/(1+np.exp(-self.Z))
    
    def tanh(self):
        '''Returns tanh actiavted results'''
        return np.tanh(self.Z)