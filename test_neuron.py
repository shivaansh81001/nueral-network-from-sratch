import numpy as np 

class test_neuron:
    def __init__(self,inputs,weights, bias):
        self.X = np.array(inputs)
        self.W= np.array(weights)
        self.b = bias

    def output(self):

        Y = np.dot(self.X,self.W) + self.b
        return Y
    

def test1():
    '''output from 4 neurons -> input to 4 neuron'''
    inputs = [1.2, 1.1, 2.4, 3.4]
    weights = [[0.3, 2.4, 3.1, 2.0],[0.3, 2.4, 0.5, 1.0],[0.3, 2.4, 3.1, 0.4],[0.3, 2.4, 2.6, 0.7]]
    bias = [1.6, 2.4,0.5, 1.2]

    neuron = test_neuron(inputs, weights, bias)
    print(neuron.output())


def test2():
    '''output from 4 neurons -> input to 1 neuron'''
    inputs = [1.2, 1.1, 2.4, 1.9]
    weights = [0.3, 0.4, 1.1, 1.2]
    bias = 2.1

    neuron = test_neuron(inputs, weights, bias)
    print(neuron.output())


def test3():
    '''output from 4 inputs -> input to 3 neurons'''
    inputs = [1.2, 1.1, 2.4, 1.9]
    weights = [[0.3, 2.4, 3.1],[0.3, 2.4, 0.5],[0.3, 2.4, 3.1],[0.3, 2.4, 2.6]]
    bias = [1.6, 2.4, 0.5]

    neuron = test_neuron(inputs, weights, bias)
    print(neuron.output())

if __name__ == '__main__':
    test1()
    test2()
    test3()