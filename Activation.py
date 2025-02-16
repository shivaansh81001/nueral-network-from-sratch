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

    def softmax(self):
        '''Returns the softmax probabilty distribution results'''
        if self.Z.ndim==1:
            num = np.exp(self.Z - np.max(self.Z))
            return num/np.sum(num)
        else:
            num = np.exp(self.Z - np.max(self.Z,axis=1, keepdims=True))
            return num/np.sum(num, axis=1, keepdims=True)
    

def test():
    inputs = np.array([-2.1,3.1,1.1])
    act = Activation(inputs)
    print("softmax = ",act.softmax())

    print("ReLU = ",act.ReLU())

    print("sigmoid = ",act.sigmoid())

    print("Tanh = ",act.tanh())

if __name__== '__main__':
    test()
