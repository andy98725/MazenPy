# ## This class is taken from the implementation in ML HW2.
# ## Since it's not the focus of the project (QLearning is),
# ## and a fairly straightforward implementation, it should
# ## have little to no bearing on the project.

import numpy as np
import random

import learner.activations as acts


class NeuralNet:

    def __init__(self, inputNodes, outputNodes, hiddenNodes, hiddenLayers, activationFunction=acts.reLu, activationDer=acts.reLuDer):
        self.activation = activationFunction
        self.activationDer = activationDer
        # Construct an array of weights at each layer
        self.weights = []
        prevLayer = inputNodes
        for _ in range(hiddenLayers):
            # width of weight matrix is the inputs from the previous layer
            # height is the output (count) of current layer
            self.weights.append(initialArray(prevLayer, hiddenNodes))
            prevLayer = hiddenNodes
        self.weights.append(initialArray(prevLayer, outputNodes))
        # Note: Base is the final element of each perceptron weight array
    
    # Take a np array input and evaluate by current weights
    def eval(self, inp):
        self.inputs = []
        self.evals = []
        for i in range(len(self.weights)):
                # Append base value to input
                inp = np.append(inp, 1)
                self.inputs.append(inp)
                # Evaluate through matrix multiplication
                inp = inp @ self.weights[i]
                self.evals.append(inp)
                # Apply activation function
                inp = self.activation(inp)    
        return inp
    
    def __backpropInitialize(self):
        self.weightDelta = []
        for i in range(len(self.weights)):
            self.weightDelta.append(zeros(len(self.weights[i]), len(self.weights[i][0])))
        
    def __backpropCase(self, inp, out, weight=1):
        evalOut = self.eval(inp)
        
        err = [None] * len(self.weights)
        # Output error of last layer
        err[-1] = (evalOut - out) * self.activationDer(self.evals[-1]) * weight
        # Outer is used because numpy matrix multiplication is awkward with 2 1D arrays
        self.weightDelta[-1] = self.weightDelta[-1] + np.outer(self.inputs[-1], err[-1])
        # Output error of each layer
        for i in reversed(range(len(err) - 1)):
            err[i] = (err[i + 1] @ np.transpose(self.weights[i + 1])) 
            # Remove the perceptron error of the base
            err[i] = np.delete(err[i], len(err[i]) - 1)
            err[i] = err[i] * self.activationDer(self.evals[i])
            self.weightDelta[i] = self.weightDelta[i] + np.outer(self.inputs[i], err[i])
    
    def __backpropFinalize(self, learningRate, totalWeight):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.weightDelta[i] * learningRate / totalWeight
        self.weightDelta = None
    
    # Train on data in array
    def backprop(self, dataArr, learningRate, iters):
        for _ in range(iters):
            self.__backpropInitialize()
            totalWeight = 0
            for elem in dataArr:
                weight = elem[2] if len(elem) > 2 else 1
                totalWeight += weight
                self.__backpropCase(elem[0], elem[1], weight)
            self.__backpropFinalize(learningRate, totalWeight)
            
    # Train in batches on data in array
    def stochasticTrain(self, dataArr, learningRate, iters, batchSize):
        for _ in range(iters):
            self.backprop(random.sample(dataArr, batchSize), learningRate, 1)
            
    # Get squared error of data
    def error(self, data):
        err = 0
        totalWeight = 0
        for elem in data:
            weight = elem[2] if len(elem) > 2 else 1
            for e in (elem[1] - self.eval(elem[0])):
                err += e * e / 2 * weight
            totalWeight += weight
        return err / totalWeight if totalWeight > 0 else 0
    
    # Utilities    
    def printWeights(self):
        for l in range(len(self.weights)):
            print("Layer {}:".format(l))
            print(self.weights[l])


    
# How the weights are initialized
def initialArray(wid, hei):
    arr = glorotArray(wid, hei)
    # Biases are all 0 initially
    arr.append([0] * hei)
    return np.array(arr)


def glorotArray(wid, hei):
    sd = np.sqrt(6.0 / (wid + hei))
    return [[np.float32(np.random.uniform(-sd, sd)) for _ in range(hei)] for _ in range(wid)]


def glorotNormArray(wid, hei):
    sd = np.sqrt(2.0 / (wid + hei))
    return [[np.float32(np.random.normal(0, sd)) for _ in range(hei)] for _ in range(wid)]


def heArray(wid, hei):
    sd = np.sqrt(2.0 / wid)
    return [[np.float32(np.random.normal(0, sd)) for _ in range(hei)] for _ in range(wid)]


# Construct a 2D array of size (wid, hei) with random values
def randomArray(wid, hei):
    return [[np.float32(np.random.uniform(-2.0, 2.0)) for _ in range(hei)] for _ in range(wid)]


# Construct a 2D array of size (wid, hei) incrementing by inc    
def spreadArray(wid, hei):
    inc = 1.0 / hei
    return [[np.float32(inc * (y + x * hei - (wid * hei - 1.0) / 2)) for y in range(hei)] for x in range(wid)]


# Construct a 2D array of size (wid, hei)
def zeros(wid, hei):
    arr = [[0 for _ in range(hei)] for _ in range(wid)]
    return np.array(arr)
