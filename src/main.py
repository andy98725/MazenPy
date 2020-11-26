from sim.maze import Maze, MAZE_SIZE
from UI.mazeDisp import MazeDisp
from learner.neuralNet import NeuralNet
from learner.QLearn import QLearn, Test
import learner.activations as acts

from os import path
import pickle


def main():
    AILoop("Sample.nn", 8, False, "Progress")
#     userLoop()
    print()
    print("Goodbye.")


# Train AI to interact with maze    
def AILoop(filename=None, iters=-1, save=False, imageFilename=None):
    # Load AI
    if filename == None:
        AI = newAI()
    elif not path.exists(filename):
        print("File {} does not exist yet.".format(filename))
        AI = newAI()
    else:
        AI = readFromFile(filename)
    
    # Main loop
    count = 1
    while iters != 0:
        print()
        print("ITERATION {}".format(count))
        print()
        # Display current progress
        Test(AI)
        
        # Train AI on some mazes
        QLearn(AI, mazeCount=128, maxIters=128, exploreRate=0.2, learningRate=0.001, learningIters=1)
        
        # Show progress briefly
        imgFile = imageFilename + str(count) + ".png"
        MazeDisp(Maze()).AILoop(AI, filename=imgFile)
        
        if iters > 0:
            iters -= 1
        count += 1
    
        # Save AI
        if save:
            writeToFile(AI, filename)


def newAI():
    inputSize = MAZE_SIZE * MAZE_SIZE  # Info for each visited node
    outputSize = 4
    hiddenSize = MAZE_SIZE * MAZE_SIZE
    hiddenLayers = 0
    
    print("Generating new AI...")
    return NeuralNet(inputSize, outputSize, hiddenSize, hiddenLayers, activationFunction=acts.sigmoid, activationDer=acts.sigmoidDer)

        
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
