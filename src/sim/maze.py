from sim import tile
import time

D_UP = 1
D_DOWN = -1
D_LEFT = 2
D_RIGHT = -2

class Maze:
    
    def __init__(self, autoTraverse = True, autoTraverseDelay = 0):
        self.size = 16
        self.autoTraverse = autoTraverse
        self.autoTraverseDelay = autoTraverseDelay
        self.tiles = tile.genMap(self.size)
        self.tilesFlat = flatten(self.tiles)
        self.currentLoc = (0, 0)
        self.traverseReady = True
        
    def currentTile(self):
        return self.tiles[self.currentLoc[0]][self.currentLoc[1]]
    def isFinished(self):
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
    
    def traverse(self, d, disp=None):
        if not self.getCurrentDirections().__contains__(d):
            return -1
        if not self.traverseReady:
            return -2
        if self.isFinished():
            return 0
        
        self.traverseReady = False
        prevTile = self.currentTile()
        if d == D_UP:
            self.currentLoc = (self.currentLoc[0], self.currentLoc[1]-1)
        elif d == D_DOWN:
            self.currentLoc = (self.currentLoc[0], self.currentLoc[1]+1)
        elif d == D_LEFT:
            self.currentLoc = (self.currentLoc[0]-1, self.currentLoc[1])
        elif d == D_RIGHT:
            self.currentLoc = (self.currentLoc[0]+1, self.currentLoc[1])
        else:
            raise Exception("Bad traversal call: d {}".format(d))
        
        prevTile.visited = not self.currentTile().visited
        if disp != None:
            disp.update()
        
        if self.autoTraverse and self.currentTile().isHallway():
            if self.autoTraverseDelay > 0:
                time.sleep(self.autoTraverseDelay)
                
            self.traverseReady = True
            return self.traverse(self.currentTile().getOtherDirection(-d), disp)
        
        else:
            self.traverseReady = True
            return 0
        
        

# Helper funct
def flatten(l):
    ret = []
    for tilelist in l:
        for tile in tilelist:
            ret.append(tile)
    return ret