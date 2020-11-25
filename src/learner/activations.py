import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDer(x):
    return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def tanhDer(x):
    return 1-(tanh(x) * tanh(x))

def reLu(x):
    return np.maximum(x, 0)

def reLuDer(x):
    return (x > 0)
    