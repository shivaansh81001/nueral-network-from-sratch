import numpy as np 

class Loss:

    def __init__(self,predicted, observed):
        self.y_hat = predicted
        self.y_target = observed
        self.N = self.y_hat.shape[0]
        self.e = 1e-10

    def cross_entropy_loss(self):
        self.y_hat = np.clip (self.y_hat, self.e, 1.0- self.e)
        denom = -(1/self.N)
        log_loss= np.sum(self.y_target * np.log(self.y_hat + self.e))
        L = denom * log_loss
        return L

    def MSE(self):
        num = len(self.y_hat)
        loss = np.sum((self.y_target - self.y_hat)**2)
        L = num*loss
        return L
    
    def binary_cross_entropy_loss(self):
        pass

    
def test():
    predicted = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
    observed = np.array([[0,0,0,1],
                        [0,0,0,1]])
    loss= Loss(predicted,observed)
    print('cross entropy multi-class= ',loss.cross_entropy_loss())

    predicted_mse = np.array([0.3, 2.4, 3.1])
    observed_mse = np.array([2.1,0.1,2.5])
    loss= Loss(predicted_mse,observed_mse)
    print('MSE= ',loss.MSE())

if __name__ == '__main__':
    test()