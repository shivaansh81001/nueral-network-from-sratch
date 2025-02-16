# nueral-network-from-sratch

## Test neuron class 

This class isolates a tensor / layer and tests for different inputs and weights. 
There are 3 tests currently ,
1. output from 4 neurons -> input to 4 neuron
2. output from 4 neurons -> input to 1 neuron
3. output from 4 inputs -> input to 3 neurons

The output is calaculated by multiplying the inputs (which is the output from previous tensor/layer) and the weights associated with the connected neurons and adds bias.
this mimics the y = <b>W<sup>-1</sup>.X + b </b>

in Numpy, for the sake of saving the computation power, I have swapped np.dot(W.T, X) to np.dot(X,W)


## Layer Class 

This class forms the hidden layer inside the neural network, 
The class is initialized with outputs from the previous layers as input (X), randomly initialized weights (W) and zero-initialized biases (B)
the class has a forward pass method which performs the <b>W<sup>-1</sup>.X + b</b> for the current layer 


## Activation Class

This class accepts Z (<b>W<sup>-1</sup>.X + b</b>) as argument contains three commonly used non-linear activation functions -> ReLU, Sigmoid, Tanh and Softmax. 



