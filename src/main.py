from sim.maze import Maze, MAZE_SIZE
from UI.mazeDisp import MazeDisp
from learner.neuralNet import NeuralNet

from os import path
import pickle

def main():
    AILoop()
    
def AILoop(filename=None, iters = -1, save = False):
    if filename == None:
        AI = newAI()
    elif not path.exists(filename):
        print("File {} does not exist yet.".format(filename))
        AI = newAI()
    else:
        AI = readFromFile(filename)
        
    while iters != 0:
        MazeDisp(Maze()).AILoop(AI)
        if iters > 0:
            iters -= 1
        
    if save:
        writeToFile(AI, filename)


def newAI():
    inputSize = 2 + MAZE_SIZE * MAZE_SIZE  # 256 for visited nodes, 2 for location
    outputSize = 4
    hiddenSize = 128
    hiddenLayers = 4
    
    print("Generating new AI...")
    return NeuralNet(inputSize, outputSize, hiddenSize, hiddenLayers)

        
def userLoop():
    while True:
        MazeDisp(Maze()).displayLoop()

# # File IO
def readFromFile(file):
    print("Loading AI from file {} ...".format(file))
    with open(file, 'rb') as inp:
        return pickle.load(inp)


def writeToFile(net, file):
    print("Writing AI to file {} ...".format(file))
    with open(file, 'wb') as outp:
        pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)


main()
