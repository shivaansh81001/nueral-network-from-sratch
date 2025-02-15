# nueral-network-from-sratch

### test neuron class 

This class isolates a tensor / layer and tests for different inputs and weights. 
There are 3 tests currently ,
1. output from 4 neurons -> input to 4 neuron
2. output from 4 neurons -> input to 1 neuron
3. output from 4 inputs -> input to 3 neurons

The output is calaculated by multiplying the inputs (which is the output from previous tensor/layer) and the weights associated with the connected neurons and adds bias.
this mimics the y = W${-1}$ * X +b 

in Numpy, for the sake of saving the computation power, I have swapped np.dot(W.T, X) to np.dot(X,W)