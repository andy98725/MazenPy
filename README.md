# About

This is a final project for CS 6375.002 (Machine Learning) at University of Texas at Dallas.

It was completed by Andy Hudson and Umar Khalid on 11/19/2020.

It uses Reinforcement Learning (through an implementation of the Q Learning algorithm) to model a maze without prior knowledge of the wall pattern.
This is done by rewarding a model for reaching the goal and punishing it for trying to move through a wall.

Though the learning algorithm is model-independent, it was tested with a Neural Net model and a Lookup Table model.
Of these, the Lookup Table model was primarily used for its speed and ease of modification.

Though it is not complete, the algorithm proves to be quite successful in solving mazes from 60-90% of the starting locations.

# Heatmaps

After each iteration, the program generates a heatmap of which cells the model can solve (starting from) and which it cannot. This demonstrates its ability to progressively learn the shape of the maze.

# Sample Runs

In the logs directory, there are 3 sample runs including info about the parameters and the heatmaps generated.

# Running

The project requires pyGame.

Run main.py with python 3.
