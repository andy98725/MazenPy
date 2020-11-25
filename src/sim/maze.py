from sim import tile
import numpy as np
import time

D_UP = 1
D_DOWN = -1
D_LEFT = 2
D_RIGHT = -2

NNDIRS = {0 : D_UP, 1 : D_LEFT, 2 : D_DOWN, 3 : D_RIGHT}
REVERSE_NNDIRS = {D_UP:0, D_LEFT:1, D_DOWN:2, D_RIGHT:3}
DIRNAMES = {D_UP : "Up", D_LEFT: "Left", D_DOWN: "Down", D_RIGHT: "Right"}

MAZE_SIZE = 16


class Maze:
    
    def __init__(self, autoTraverse=True, autoTraverseDelay=0, maxTraversals=-1):
        self.size = MAZE_SIZE
        self.autoTraverse = autoTraverse
        self.autoTraverseDelay = autoTraverseDelay
        self.maxTraversals = maxTraversals
        self.traversals = 0
        
        self.tiles = tile.genMap(self.size)
        self.tilesFlat = flatten(self.tiles)
        
        self.currentLoc = (0, 0)
        self.traverseReady = True
        
    def currentTile(self):
        return self.tiles[self.currentLoc[0]][self.currentLoc[1]]

    def isFinished(self):
        return self.success() or (self.maxTraversals > 0 and self.traversals >= self.maxTraversals)
    
    def success(self):
        return self.currentTile().end
        
    def getCurrentDirections(self):
        # Get current tile occupied
        tile = self.currentTile()
        
        dirs = []
        if tile.up:
            dirs.append(D_UP)
        if tile.down:
            dirs.append(D_DOWN)
        if tile.left:
            dirs.append(D_LEFT)
        if tile.right:
            dirs.append(D_RIGHT)
        return dirs        
    
    def traverse(self, d, disp=None, recursive=False):
        if not recursive:
            self.traversals += 1
        if not self.getCurrentDirections().__contains__(d):
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
            self.currentLoc = (self.currentLoc[0], self.currentLoc[1] - 1)
        elif d == D_DOWN:
            self.currentLoc = (self.currentLoc[0], self.currentLoc[1] + 1)
        elif d == D_LEFT:
            self.currentLoc = (self.currentLoc[0] - 1, self.currentLoc[1])
        elif d == D_RIGHT:
            self.currentLoc = (self.currentLoc[0] + 1, self.currentLoc[1])
        else:
            raise Exception("Bad traversal call: d {}".format(d))
        
        prevTile.visited = not self.currentTile().visited
        if disp != None:
            disp.update()
        
        if self.autoTraverse and self.currentTile().isHallway():
            if self.autoTraverseDelay > 0:
                time.sleep(self.autoTraverseDelay)
                
            self.traverseReady = True
            return self.traverse(self.currentTile().getOtherDirection(-d), disp, True)
        
        else:
            self.traverseReady = True
            return 0
        
    def getRandomDir(self):
        dirs = self.currentTile().getDirections()
        np.random.shuffle(dirs)
        return dirs[0]
    
    def getAIDir(self, model):
        # Find highest validated direction
        validDirs = self.currentTile().getDirections()
        e = model.eval(self.getNNState())
        maxVal = 0
        maxDir = -1
        
        for i in range(len(e)):
            d = NNDIRS.get(i)
            if d in validDirs and e[i] > maxVal:
                maxDir = d
                maxVal = e[i]
        return maxDir
        
    def pilot(self, model, disp=None, pauseTime=0):
        d = self.getAIDir(model)
        
        # Delay
        if pauseTime > 0:
            time.sleep(pauseTime)
        
#         print("Input {}".format(self.getNNState()))
#         print("True output {}".format(model.eval(self.getNNState())))
#         print("Chose dir {}".format(DIRNAMES.get(d)))    
        self.traverse(d, disp)

    # Get NNet input in np array format
    def getNNState(self):
        inp = []
        
        # (0-255): Has the tile been reached? (Y/N)
        for tile in self.tilesFlat:
            if tile.visited:
                inp.append(1)
            else:
                inp.append(0)
        
        # (256, 257): Current location in maze
        inp.append(self.currentLoc[0])
        inp.append(self.currentLoc[1])
        
        return inp
    
    # Get reward for current state
    def getReward(self, force=False):
        if self.success():
            return 1.0
        return -0.05
#         if self.action == 'invalid' and not force:
#             return -1.0
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
