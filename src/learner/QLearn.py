import numpy as np
import random
import time

from sim.maze import Maze, REVERSE_NNDIRS


def Test(model, mazeCount=32, maxIters=64):
    totalSteps = 0
    totalSolved = 0
    
    for _ in range(mazeCount):
        maze = Maze(maxTraversals=maxIters)
        
        while not maze.isFinished():
            maze.traverse(maze.getAIDir(model))
        
        if maze.success():
            totalSteps += maze.traversals
            totalSolved += 1
    
    print("AI solved {} out of {} mazes in {} avg steps".format(totalSolved, mazeCount, totalSteps / totalSolved if totalSolved > 0 else "-"))


def QLearn(model, mazeCount, maxIters, exploreRate, learningRate, learningIters):
    print("Training AI on {} mazes, {} max steps with explore rate {}".format(mazeCount, maxIters, exploreRate))
    print("Using learning rate {} and {} GD iters at each step".format(learningRate, learningIters))
    
    memory = QLearnMem()
    startTime = time.time()
    
    for i in range(mazeCount):
        maze = Maze(maxTraversals=maxIters)
        goodActs = 0
        badActs = 0
        invalidActs = 0
        minDist = maze.currentTile().distance
        prevDist = minDist
        
        while not maze.isFinished():
            # Get action according to exploration rate
            if np.random.rand() < exploreRate:
                act = maze.getRandomDir()
            else:
                act = maze.getAIDir(model)
            
            # Get action information and progress state
            curState = maze.getNNState()
            maze.traverse(act)
            nextState = maze.getNNState()
            reward = maze.getReward()
            
            # Track actions
            newDist = maze.currentTile().distance
            if newDist < minDist:
                goodActs += 1
                minDist = newDist
            elif newDist == prevDist:
                invalidActs += 1
            else:
                badActs += 1
                
            prevDist = newDist
            
            # Store info in memory
            memory.append(curState, REVERSE_NNDIRS.get(act), reward, nextState, maze.isFinished())
            
            # Learn from current memory state
            model.backprop(memory.getTrainingData(model), learningRate, learningIters)
        print("({} seconds) Maze {} finished with {} good, {} bad, {} invalid actions, {} tiles away".format(round(time.time() - startTime, 1), i + 1, goodActs, badActs, invalidActs, maze.getDistance()))


class QLearnMem:

    def __init__(self, memSize=100, discount=1):
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
            if gameOver:
                output[act] = reward
            else:
                nextReward = np.max(model.eval(result))
                output[act] = reward + self.discount * nextReward   

            dataPairs.append((inp, output))
        
        random.shuffle(dataPairs)
        return dataPairs
