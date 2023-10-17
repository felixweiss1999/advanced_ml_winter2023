import numpy as np
import types

class FeedforwardNet:
    def __init__(self, m : int, n : int, activation_function : types.FunctionType, weight_matrices : np.array = None, biases : np.array = None):
        
        self.m = m
        self.n = n
        self.a = activation_function
        self.biases = biases
        self.weight_matrices = weight_matrices
        assert len(self.biases) == self.m-1
        assert len(self.weight_matrices) == self.m-1
        
    def __call__(self, x):
        for i in range(0, self.m-1):
             x = self.a(self.weight_matrices[i] @ x + self.biases[i])
        return x
        

andNet = FeedforwardNet(n=1,m=2, activation_function=lambda x : [max(0, i) for i in x], weight_matrices=np.array([[[1, 1]]]), biases=np.array([[-1]]))


w_1 = np.array([[-1, -1],\
                [-1,  1],\
                [ 1, -1],\
                [ 1,  1]])
b_1 = np.array([1,\
                0,\
                0,\
               -1])
testnet =  FeedforwardNet(n=1,m=2, activation_function=lambda x : [max(0, i) for i in x], weight_matrices=np.array([w_1]), biases=np.array([b_1]))

testnet_and = FeedforwardNet(n=1,m=3, activation_function=lambda x : [max(0, i) for i in x], weight_matrices=[w_1, np.array([0,0,0,1])], biases=[b_1, np.array([0])])
testnet_or = FeedforwardNet(n=1,m=3, activation_function=lambda x : [max(0, i) for i in x], weight_matrices=[w_1, np.array([0,1,1,1])], biases=[b_1, np.array([0])])
testnet_xor = FeedforwardNet(n=1,m=3, activation_function=lambda x : [max(0, i) for i in x], weight_matrices=[w_1, np.array([0,1,1,0])], biases=[b_1, np.array([0])])
print(testnet_or([1,0]))