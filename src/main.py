from sim.maze import Maze
from UI.mazeDisp import MazeDisp
# from learner.models.neuralNet import NeuralNet
# import learner.models.activations as acts
from learner.models.DecisionTable import DecisionTable
from learner.QLearn import QLearn, Test

from os import path
import pickle


def main():
    AILoop("Med.dict", "Med.maze", 16, False, "SampleRun 3 ")
#     userLoop("Small.maze")
    print()
    print("Goodbye.")


# Train AI to interact with maze    
def AILoop(filename=None, mazeFilename=None, iters=-1, save=False, imageFilename=None):
    AI = getAI(filename)
    maze = getMaze(mazeFilename)
    
    if save and mazeFilename != None:
        writeToFile(maze, mazeFilename)
    
    # Main loop
    count = 1
    while iters != 0:
        print()
        print("ITERATION {}".format(count))
        print()
        # Display current progress
        Test(AI, maze)
        
        # Train AI on some mazes
        QLearn(AI, maze, maxSteps=maze.longestPath)
        
        # Show progress briefly
        if imageFilename == None:
            imgFile = None
        else:
            imgFile = imageFilename + str(count) + ".png"
        maze.reset()
        maze.maxTraversals = maze.longestPath
        MazeDisp(maze).AILoop(AI, filename=imgFile)
        
#         AI.printState()
        
        if iters > 0:
            iters -= 1
        count += 1
    
        # Save AI
        if save and filename != None:
            writeToFile(AI, filename)
            

def getAI(filename=None):
    if filename == None:
        return newAI()
    elif not path.exists(filename):
        print("File {} does not exist yet.".format(filename))
        return newAI()
    else:
        return readFromFile(filename)


def newAI():
    print("Generating new AI...")
#     inputSize = MAZE_SIZE * MAZE_SIZE  # Info for each visited node
#     outputSize = 4
#     hiddenSize = MAZE_SIZE * MAZE_SIZE
#     hiddenLayers = 0
#     
#     return NeuralNet(inputSize, outputSize, hiddenSize, hiddenLayers, activationFunction=acts.sigmoid, activationDer=acts.sigmoidDer)
    return DecisionTable(4)


def getMaze(filename=None):
    if filename == None:
        return newMaze()
    elif not path.exists(filename):
        print("File {} does not exist yet.".format(filename))
        return newMaze()
    else:
        return readFromFile(filename)


def newMaze():
    print("Generating new maze...")
    return Maze()

        
def userLoop(mazeFilename=None):
    while True:
        # Use 
        MazeDisp(getMaze(filename=mazeFilename)).displayLoop()


# # File IO
def readFromFile(file):
    print("Loading from file {} ...".format(file))
    with open(file, 'rb') as inp:
        return pickle.load(inp)


def writeToFile(obj, file):
    print("Writing to file {} ...".format(file))
    with open(file, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


main()
