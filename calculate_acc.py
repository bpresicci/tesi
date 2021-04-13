import numpy as np

def accuracy(accuracies):
    mean = np.mean(accuracies, 1)
    var = np.var(accuracies, 1)
    return mean, var