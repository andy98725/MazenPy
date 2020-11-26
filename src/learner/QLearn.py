import numpy as np
import random
import time

from sim.maze import NNDIRS


def eval(model, maze):
    return np.argmax(model.eval(maze.getState()))


def Test(model, maze, mazeCount=32, maxIters=64):
    maze.maxTraversals = maxIters
    
    totalSteps = 0
    totalSolved = 0
    for _ in range(mazeCount):
        maze.reset(True)
        
        while not maze.isFinished():
            maze.traverse(NNDIRS.get(eval(model, maze)))
        
        if maze.success():
            totalSteps += maze.traversals
            totalSolved += 1
    
    print("AI solved {} out of {} mazes in {} avg steps".format(totalSolved, mazeCount, totalSteps / totalSolved if totalSolved > 0 else "-"))


def QLearn(model, maze, **options):
    mazeCount = options.get('iters', 12)
    maxSteps = options.get('maxSteps', maze.longestPath)
    exploreRate = options.get('exploration', 0.1)
    
    memSize = options.get('memSize', 16 * 16 * 8)
    memDiscount = options.get('memDiscount', 0.1)
    batchSize = options.get('batchSize', 256)
    learnerRate = options.get('learningRate', 0.8)
    learnerIters = options.get('batchIters', 1)
    
    print("Training AI {} times, {} max steps with explore rate {}".format(mazeCount, maxSteps, exploreRate))
    print("Using learning rate {}, batch size {}, and {} GD iters at each step".format(learnerRate, batchSize, learnerIters))
    
    maze.maxTraversals = maxSteps
    startTime = time.time()
    memory = QLearnMem(memSize, memDiscount)
    
    for i in range(mazeCount):
        maze.reset()
        
        exploreActs = 0
        backtrackActs = 0
        invalidActs = 0
        
        initLoc = maze.loc
        initDist = maze.currentTile().distance
        
        while not maze.isFinished():
            # Get action according to exploration rate
            if np.random.rand() < exploreRate:
                act = np.random.randint(4)
            else:
                act = eval(model, maze)

            # Get action information and progress state
            curState = maze.getState()
            maze.traverse(NNDIRS.get(act))
            nextState = maze.getState()
            reward = maze.getReward()
            
            # Track actions
            if maze.state == 'invalid':
                invalidActs += 1
            elif maze.exploredTile():
                backtrackActs += 1
            else:
                exploreActs += 1
            
            # Store info in memory
            memory.append(curState, act, reward, nextState, maze.success() or maze.failure())
            
            # Learn from current memory state
#             model.stochasticFit(memory.getTrainingData(model), learnerRate, learnerIters, batchSize)
            
            model.fit(memory.getTrainingData(model), learnerRate, learnerIters)
        
#         mse = model.error(memory.getTrainingData(model))
#         print("Iter {} | {} s | Finished with {} explore, {} backtrack, {} blocked actions".format(i + 1, round(time.time() - startTime, 1), exploreActs, backtrackActs, invalidActs))
#         print("Started {} (dist {}), ended {} (dist {})".format(initLoc, initDist, maze.loc, maze.getDistance()))
#         print("MSE {}".format(mse))
    mse = model.error(memory.getTrainingData(model))
    print("({} sec) Final MSE {}".format(round(time.time() - startTime, 1), mse))


class QLearnMem:

    def __init__(self, memSize=10, discount=1):
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
