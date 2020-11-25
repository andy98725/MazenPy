import numpy as np
import random
import time

from sim.maze import Maze, REVERSE_NNDIRS


def QLearn(model, mazeCount=12, maxIters=64, exploreRate=0.1, learningRate=0.0001, learningIters=1):
    print("Training AI on {} mazes, {} max iters with explore rate {}".format(mazeCount, maxIters, exploreRate))
    print("Using learning rate {} and {} GD iters at each step".format(learningRate, learningIters))
    
    memory = QLearnMem()
    startTime = time.time()
    
    for i in range(mazeCount):
        maze = Maze(maxTraversals=maxIters)
        actionsTaken = 0
        
        while not maze.isFinished():
            # Get action according to exploration rate
            if np.random.rand() < exploreRate:
                act = maze.getRandomDir()
            else:
                act = maze.getAIDir(model)
            
            # Get action information and progress state
            curState = maze.getNNState()
            maze.traverse(act)
            actionsTaken += 1
            
            nextState = maze.getNNState()
            reward = maze.getReward()
            
            # Store info in memory
            memory.append(curState, REVERSE_NNDIRS.get(act), reward, nextState, maze.isFinished())
            
            # Learn from current memory state
            model.backprop(memory.getTrainingData(model), learningRate, learningIters)
        
        print("({} seconds) Maze {} finished in {} actions with final distance from exit of {}".format(round(time.time() - startTime, 1), i, actionsTaken, round(maze.getDistance(), 3)))


class QLearnMem:

    def __init__(self, memSize=100, discount=0.8):
        self.size = memSize
        self.discount = discount
        self.memory = []
    
    # Events are stored as [state, action, reward, result, gameOver]
    def append(self, state, act, reward, resultState, gameOver):
        self.memory.append((state, act, reward, resultState, gameOver))
        if len(self.memory) > self.size:
            del self.memory[0]
    
    def getTrainingData(self, model):
        dataPairs = []
        for (state, act, reward, result, gameOver) in self.memory:
            # The input is just the initial state
            inp = state
            # The output should only change for the action
            output = model.eval(state)
            output[act] = reward
            
            # Factor in Q_sa 
            if not gameOver:
                futureReward = model.eval(result)
                output[act] += np.max(futureReward)

            dataPairs.append((inp, output))
        
        random.shuffle(dataPairs)
        return dataPairs
