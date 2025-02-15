import numpy as np 

class test_neuron:
    def __init__(self,inputs,weights, bias):
        self.X = np.array(inputs)
        self.W= np.array(weights)
        self.b = bias

    def output(self):

        Y = np.dot(self.X,self.W) + self.b
        return Y
    
    