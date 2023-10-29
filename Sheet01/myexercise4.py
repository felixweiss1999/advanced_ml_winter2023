import numpy as np
class ReLU:
    def __init__(self):
        pass
    def __call__(self, x):
        return [max(0, i) for i in x]
instance = ReLU()
print(instance(np.array([-1, 0, 1, -3, 2])))