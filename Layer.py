import numpy as np

np.random.seed(0)

class Layer():
    def __init__(self, inputs, n_neurons):
        '''class accepts inputs (outputs from the previous layers) and the number of required neurons in the current layer'''
        self.X = inputs
        self.W = 0.1* np.random.randn(len(self.X),n_neurons)
        self.B = np.array([0 for i in range(n_neurons)])

    def forward_pass(self):
        '''method performs the necesary matrix multiplications'''
        print(self.W, self.X, self.B)
        return np.dot(self.X,self.W) + self.B
    

def test_forward():
    #testing method for forward pass 
    inputs = [1.2, 1.1, 2.4, 3.4]
    tensor = Layer(inputs, 4)
    print(tensor.forward_pass())
    

if __name__=='__main__':
    test_forward()


    