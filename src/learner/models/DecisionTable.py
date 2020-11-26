import numpy as np
import random


class DecisionTable:

    def __init__(self, actionLen):
        self.actionLen = actionLen
        self.dict = dict()
        
    def printState(self):
        print(self.dict)
        
    def eval(self, inp):
        inp = tuple(inp)
        
        if not inp in self.dict:
            # Random 0-1 initialization
#             self.dict[inp] = np.random.rand(self.actionLen)
            self.dict[inp] = np.zeros(self.actionLen)
        
        return np.array(self.dict[inp])
    
    def fit(self, dataArr, learningRate, iters, maxError=None):
        for [inp, outp] in dataArr:
            self.dict[inp] = np.array(outp)
#         while True:
#             for _ in range(iters):
#                 changeDict = dict()
#                  
#                 for [inp, outp] in dataArr:
#                     inp = tuple(inp)
#                     if not inp in changeDict:
#                         changeDict[inp] = np.zeros(self.actionLen)
#                      
#                     err = outp - self.eval(inp)
#                     changeDict[inp] += err
#      
#                 for key in changeDict:
#                     self.dict[key] += learningRate * changeDict[key] / len(dataArr)
#             # Escape check
#             if maxError == None or self.error(dataArr) < maxError:
#                 break;
#      
    def stochasticFit(self, dataArr, learningRate, iters, batchSize, maxError=None):
        batchSize = min(batchSize, len(dataArr))
        for _ in range(iters):
            self.fit(random.sample(dataArr, batchSize), learningRate, 1, maxError)
        
        
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