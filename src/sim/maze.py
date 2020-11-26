import time

import numpy as np
from sim import tile

D_UP = 1
D_DOWN = -1
D_LEFT = 2
D_RIGHT = -2

NNDIRS = {0 : D_UP, 1 : D_LEFT, 2 : D_DOWN, 3 : D_RIGHT}
REVERSE_NNDIRS = {D_UP:0, D_LEFT:1, D_DOWN:2, D_RIGHT:3}
DIRNAMES = {D_UP : "Up", D_LEFT: "Left", D_DOWN: "Down", D_RIGHT: "Right"}

MAZE_SIZE = 16

# Default/consistent maze
DEF_MAP = tile.genMap(MAZE_SIZE)


class Maze:
    
    def __init__(self, autoTraverse=False, autoTraverseDelay=0, maxTraversals=-1, allowBacktracking=True):
        self.size = MAZE_SIZE
        self.autoTraverse = autoTraverse
        self.autoTraverseDelay = autoTraverseDelay
        self.maxTraversals = maxTraversals
        self.traversals = 0
        self.visited = []
        self.allowBacktracking = allowBacktracking
        
        self.tiles, self.longestPath = tile.genMap(self.size)
#         self.tiles = tile.copyMap(DEF_MAP, randomizeStart=True)
        self.tilesFlat = flatten(self.tiles)
        
        self.loc = self.startLoc()
        self.traverseReady = True
        
    def reset(self, changeStart=True, startTile=None):
        self.traverseReady = False
        
        self.traversals = 0
        self.state = None
        self.visited = []
        for tile in self.tilesFlat:
            tile.visited = False
        
        if changeStart:
            for tile in self.tilesFlat:
                tile.start = False
                
            if startTile != None:
                startTile.start = True
            else:  # randomized
                index = np.random.randint(len(self.tilesFlat) - 1)
                self.tilesFlat[index].start = True
            
        self.loc = self.startLoc()
        self.traverseReady = True
    
    def startLoc(self):
        for t in self.tilesFlat:
            if t.start:
                return (t.x, t.y)
        
    def currentTile(self):
        return self.tiles[self.loc[0]][self.loc[1]]

    def isFinished(self):
        return self.success() or self.failure() or self.timeOut()

    def success(self):
        return self.currentTile().end

    def failure(self):
        return not self.allowBacktracking and len(self.getCurrentDirections()) == 0 and not self.success()

    def timeOut(self):
        return self.maxTraversals > 0 and self.traversals >= self.maxTraversals
        
    def getCurrentDirections(self):
        # Get current tile occupied
        tile = self.currentTile()
        
        dirs = []
        if tile.up and (self.allowBacktracking or not self.tiles[self.loc[0]][self.loc[1] - 1].visited):
            dirs.append(D_UP)
        if tile.down and (self.allowBacktracking or not self.tiles[self.loc[0]][self.loc[1] + 1].visited):
            dirs.append(D_DOWN)
        if tile.left and (self.allowBacktracking or not self.tiles[self.loc[0] - 1][self.loc[1]].visited):
            dirs.append(D_LEFT)
        if tile.right and (self.allowBacktracking or not self.tiles[self.loc[0] + 1][self.loc[1]].visited):
            dirs.append(D_RIGHT)
        return dirs        
    
    def getOtherDirection(self, d):
        dirs = self.getCurrentDirections()
        if d in dirs:
            dirs.remove(d)
        return dirs[0]
    
    def traverse(self, d, disp=None, recursive=False):
        finished = self.isFinished()
        if not recursive:
            self.traversals += 1
        if not d in self.getCurrentDirections():
            # Store self state for training
            self.state = 'invalid'
            return -1
        if not self.traverseReady:
            return -2
        if finished:
            return -3
        self.state = 'valid'
        
        self.traverseReady = False
        prevTile = self.currentTile()
        if d == D_UP:
            self.loc = (self.loc[0], self.loc[1] - 1)
        elif d == D_DOWN:
            self.loc = (self.loc[0], self.loc[1] + 1)
        elif d == D_LEFT:
            self.loc = (self.loc[0] - 1, self.loc[1])
        elif d == D_RIGHT:
            self.loc = (self.loc[0] + 1, self.loc[1])
        else:
            raise Exception("Bad traversal call: d {}".format(d))
        
        prevTile.visited = not self.currentTile().visited
        if prevTile not in self.visited:
            self.visited.append(prevTile)
        
        if disp != None:
            disp.update()
        
        if self.autoTraverse and self.currentTile().isHallway():
            if self.autoTraverseDelay > 0:
                time.sleep(self.autoTraverseDelay)
                
            self.traverseReady = True
            return self.traverse(self.getOtherDirection(-d), disp, True)
        
        else:
            self.traverseReady = True
            return 0
        
    def pilot(self, model, disp=None, pauseTime=0):
        
        d = NNDIRS.get(np.argmax(model.eval(self.getState())))
#         print(DIRNAMES.get(d))
        
        res = self.traverse(d, disp)
        
        # Delay
        if pauseTime > 0 and res >= 0:
            time.sleep(pauseTime)
            
    def getSolveableStates(self, model):
        solveStates = []
        for x in range(self.size):
            for y in range(self.size):
                if self.tiles[x][y].end:
                    continue
                
                self.reset(True, self.tiles[x][y])
                while not self.isFinished():
                    self.pilot(model)
                
                if self.success():
                    solveStates.append((x, y))
        return solveStates
    
    def clearModelSuccessStates(self):
        for tile in self.tilesFlat:
            tile.softSuccess = False
            tile.softFail = False
        
    def evalModelSuccessStates(self, model, maxTraversals=64):
        self.clearModelSuccessStates()
        self.maxTraversals = maxTraversals
        solvedStates = self.getSolveableStates(model)
        
        for x in range(self.size):
            for y in range(self.size):
                if self.tiles[x][y].end:
                    continue
                
                if (x, y) in solvedStates:
                    self.tiles[x][y].softSuccess = True
                else:
                    self.tiles[x][y].softFail = True

    # Get NNet input in np array format
    def getState(self):
        inp = []
        
#         for tile in self.tilesFlat:
#             if tile == self.currentTile():
#                 inp.append(1)
# #             elif tile.visited:
# #                 inp.append(-1)
#             else:
#                 inp.append(0)
        inp.append(self.currentTile().x)
        inp.append(self.currentTile().y)
        
        return tuple(inp)
    
    # Get reward for current state
    def getReward(self):
        if self.success():
            return 100.0

#         if self.currentTile().deadEnd():
#             return -100.0
        # Illegal move
        if self.state == 'invalid':
            return -1

#         # Non-exploratory move        
#         if self.exploredTile():
#             return -0.25

        return -0.1
    
    def exploredTile(self):
        return self.currentTile() in self.visited
    
    def getDistance(self):
        return self.currentTile().distance

            
# Helper funct
def flatten(l):
    ret = []
    for tilelist in l:
        for tile in tilelist:
            ret.append(tile)
    return ret
