import numpy as np

def accuracy_axis_1(accuracies):
    mean = np.mean(accuracies, 1)
    var = np.var(accuracies, 1)
    return mean, var
def accuracy_axis_2(accuracies):
    mean = np.mean(accuracies, 2)
    var = np.var(accuracies, 2)
    return mean, var