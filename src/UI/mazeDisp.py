from sim import tile
from sim import maze

import pygame
pygame.init()


class MazeDisp:

    def __init__(self, maze):
        self.maze = maze
        self.size = tile.PX_SIZE * maze.size
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.running = True
        
        maze.autoTraverseDelay = 0.02

    # Draw self and scan for events
    def displayLoop(self):
        self.update()
        
        while self.running:
            for event in pygame.event.get():
                # User clicked X
                if event.type == pygame.QUIT:
                    self.quit()
                    
                # Handle button inputs
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    # Soft quit
                    elif event.key == pygame.K_SPACE:
                        self.running = False
                    
                    elif event.key == pygame.K_LEFT:
                        self.maze.traverse(maze.D_LEFT, self)
                    elif event.key == pygame.K_RIGHT:
                        self.maze.traverse(maze.D_RIGHT, self)
                    elif event.key == pygame.K_UP:
                        self.maze.traverse(maze.D_UP, self)
                    elif event.key == pygame.K_DOWN:
                        self.maze.traverse(maze.D_DOWN, self)
                        
        pygame.quit()
                        
    def AILoop(self, AI, pause=0.5, filename=None):
        self.update()
        
        while self.running:
            for event in pygame.event.get():
                # User clicked X
                if event.type == pygame.QUIT:
                    self.running = False
                    
                # Handle button inputs
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Soft quit
                    elif event.key == pygame.K_SPACE:
                        self.running = False
            
            # AI nav
            self.maze.pilot(AI, self, pause)
            
            if self.maze.isFinished():
                self.running = False
        
        # Save final state
        if filename != None:
            pygame.image.save(self.screen, filename)
            
        pygame.quit()
        
    # Redraw self
    def update(self):
        for tile in self.maze.tilesFlat:
            tile.drawBG(self.screen, self.maze.loc)
        for tile in self.maze.tilesFlat:
            tile.drawWalls(self.screen)
            
        pygame.display.update()
    
    # Quit app
    def quit(self):
        pygame.quit()
        exit()

