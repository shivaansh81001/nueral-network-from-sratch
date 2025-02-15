import numpy as np 

class test_neuron:
    def __init__(self,inputs,weights, bias):
        self.X = np.array(inputs)
        self.W= np.array(weights)
        self.b = bias

    def output(self):

        Y = np.dot(self.X,self.W) + self.b
        return Y
    

def test():
    inputs = [1.2, 1.1, 2.4]
    weights = [0.3, 2.4, 3.1]
    bias = 1.6

    neuron = test_neuron(inputs, weights, bias)
    print(neuron.output())


if __name__ == '__main__':
    test()