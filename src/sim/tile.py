import random
import pygame

from sim import maze

PX_SIZE = 32
WALL_SIZE = 4

COL_BG = (32, 32, 32)
COL_VISITED = (128, 128, 128)
COL_START = (0, 255, 0)
COL_END = (255, 0, 0)
COL_PLAYER = (255, 255, 0)
COL_WALL = (223, 223, 223)


class Tile:
    
    def __init__(self, x, y, isStart=False, isEnd=False):
        self.x = x
        self.y = y
        self.visited = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.start = isStart
        self.end = isEnd
        self.neighborCount = 0
    
    def link(self, other):
        self.neighborCount += 1
        other.neighborCount += 1
        if self.x < other.x:
            self.right = True
            other.left = True
        elif self.x > other.x:
            self.left = True
            other.right = True
        elif self.y < other.y:
            self.down = True
            other.up = True
        elif self.y > other.y:
            self.up = True
            other.down = True 
        else:
            raise Exception("Bad link call on {} to {}".format(self, other))
    
    def isHallway(self):
        return self.neighborCount == 2
    
    def setVisited(self, visited):
        self.visited = visited
        
    def getDirections(self):
        dirs = []
        if self.up:
            dirs.append(maze.D_UP)
        if self.down:
            dirs.append(maze.D_DOWN)
        if self.left:
            dirs.append(maze.D_LEFT)
        if self.right:
            dirs.append(maze.D_RIGHT)
        
        return dirs
        
    def getOtherDirection(self, d):
        dirs = self.getDirections()
        dirs.remove(d)
        return dirs[0]
        
    # Display the background to screen
    def drawBG(self, screen, currentLoc=None):
        if self.start:
            bgcol = COL_START
        elif self.end:
            bgcol = COL_END
        elif currentLoc != None and currentLoc[0] == self.x and currentLoc[1] == self.y:
            bgcol = COL_PLAYER
        elif self.visited:
            bgcol = COL_VISITED
        else:
            bgcol = COL_BG
            
        tlx = self.x * PX_SIZE
        tly = self.y * PX_SIZE
        pygame.draw.rect(screen, bgcol, (tlx, tly, PX_SIZE, PX_SIZE))
        
    # Display this tile's walls to screen
    def drawWalls(self, screen, currentLoc=None):
        tlx = self.x * PX_SIZE
        tly = self.y * PX_SIZE
        
        # Draw walls
        if not self.up:
            pygame.draw.rect(screen, COL_WALL, (tlx - WALL_SIZE / 2, tly - WALL_SIZE / 2, PX_SIZE + WALL_SIZE, WALL_SIZE))
        if not self.left:
            pygame.draw.rect(screen, COL_WALL, (tlx - WALL_SIZE / 2, tly - WALL_SIZE / 2, WALL_SIZE, PX_SIZE + WALL_SIZE))
        if not self.down:
            pygame.draw.rect(screen, COL_WALL, (tlx - WALL_SIZE/2, tly + PX_SIZE - WALL_SIZE/2, PX_SIZE+WALL_SIZE, WALL_SIZE))
        if not self.right:
            pygame.draw.rect(screen, COL_WALL, (tlx + PX_SIZE - WALL_SIZE/2, tly-WALL_SIZE/2, WALL_SIZE, PX_SIZE+WALL_SIZE))


# Generate maze
def genMap(size):
    # 2D array that will hold the tiles
    tilemap = [[None for _ in range(size)] for _ in range(size)]
    start = (0, 0)
    end = (size - 1, size - 1)
    
    # Maze generation can work as a randomized DFS
    DFS(size, tilemap, start, end, start)
    return tilemap


def DFS(size, tilemap, startLoc, endLoc, loc, prevTile=None):
    x = loc[0]
    y = loc[1]
    isStart = loc == startLoc
    isEnd = loc == endLoc
    tile = tilemap[x][y] = Tile(x, y, isStart, isEnd)
    if prevTile != None:
        tile.link(prevTile)
        
    # Force end hallway on end tile
    if isEnd:
        return
    
    # Find cases to search
    recursiveCases = []
    if y > 0 and tilemap[x][y - 1] == None:
        recursiveCases.append((x, y - 1)) 
    if y < size - 1 and tilemap[x][y + 1] == None:
        recursiveCases.append((x, y + 1))
    if x > 0 and tilemap[x - 1][y] == None:
        recursiveCases.append((x - 1, y)) 
    if x < size - 1 and tilemap[x + 1][y] == None:
        recursiveCases.append((x + 1, y))
    # Ensure random ordering for randomized paths  
    random.shuffle(recursiveCases)
    
    for nextLoc in recursiveCases:
        if tilemap[nextLoc[0]][nextLoc[1]] == None:
            DFS(size, tilemap, startLoc, endLoc, nextLoc, tile)
    
