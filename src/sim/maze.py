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
    
    def __init__(self, autoTraverse=True, autoTraverseDelay=0, maxTraversals=-1, allowBacktracking=False):
        self.size = MAZE_SIZE
        self.autoTraverse = autoTraverse
        self.autoTraverseDelay = autoTraverseDelay
        self.maxTraversals = maxTraversals
        self.traversals = 0
        self.allowBacktracking = allowBacktracking
        
#         self.tiles = tile.genMap(self.size, allowBacktracking)
        self.tiles = tile.copyMap(DEF_MAP)
        self.tilesFlat = flatten(self.tiles)
        
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
        if not recursive:
            self.traversals += 1
        if not d in self.getCurrentDirections():
#             self.action = 'invalid'
            return -1
        if not self.traverseReady:
            return -2
        if self.isFinished():
            return 0
#         self.action = 'valid'  # Store validity state for training
        
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
        
    def getRandomDir(self):
        dirs = self.getCurrentDirections()
        np.random.shuffle(dirs)
        return dirs[0]
    
    def getAIDir(self, model):
        # Find highest validated direction
        validDirs = self.getCurrentDirections()
        e = model.eval(self.getNNState())
        maxVal = -1
        maxDir = 0
        
        for i in range(len(e)):
            d = NNDIRS.get(i)
            if d in validDirs and e[i] > maxVal:
                maxDir = d
                maxVal = e[i]
        return maxDir
        
    def pilot(self, model, disp=None, pauseTime=0):
        d = self.getAIDir(model)
#         print(DIRNAMES.get(d))
        
        # Delay
        if pauseTime > 0:
            time.sleep(pauseTime)
        
        self.traverse(d, disp)

    # Get NNet input in np array format
    def getNNState(self):
        inp = []
        
        for tile in self.tilesFlat:
            if tile == self.currentTile():
                inp.append(1)
#             elif tile.visited:
#                 inp.append(-1)
            else:
                inp.append(0)
        
        return inp
    
    # Get reward for current state
    def getReward(self):
        if self.success():
            return 1.0
        if self.failure():
            return -1.0
        return -0.05
#         if self.isFinished():
#             return self.currentTile().reward
#         if self.action == 'invalid':
#             return -0.4
        # Get distance from finishing
#         return self.currentTile().reward
    
    def getDistance(self):
        return self.currentTile().distance

            
# Helper funct
def flatten(l):
    ret = []
    for tilelist in l:
        for tile in tilelist:
            ret.append(tile)
    return ret
