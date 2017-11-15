# image_GA
image mimicry genetic algorithm

# Specification

## Algorithm Flow
- Generate Random Population, computing fitness for each member
- Until Stopping Condition:
    - pick two random individuals
    - Do crossover to produce two children
    - mutate each child with some probability
    - recompute fitness for children
    - Add children back to Population
    - Drop two least fit members of Population
    - Enqueue fitness of top individual, dequeueing oldest fitness if Queue beyond max length.

## "Global" Variables (these might get absorbed into classes)
- Image Resolution / bitmap dimensions / x & y range
- Population size
- Number of vertices per shape
- Number of shapes per individual
- Mutation probability for new children
- Mutation probability per shape in individual
- Stopping test tolerance (if difference between oldest and newest best-fitness is lower than this, stop)
- Max number of generations/cycles to perform
- Max queue length (for fitness queue)
- Starting image
- Color range

## Classes

### Individual

shapes: [shape]

fitness: Float

image/bitmap: [[(int, int, int)]]

random\_init()

fitness()

image\_render()

mutate()

### Shape

vertices: [(int, int)]

RGBA: (int, int, int, float)  Note: float value in closed interval [0.0,1.0]

mutate()

### Population

members: [Individual]

random\_init()

crossover()

# Tasks

## Class Wireframes

Tanner

## Image Rendering

Sukhdeep

## Crossover

Lakshmi

## Mutation

Justin

## Fitness

Daniel

## Population Generation

Tanner

## Unit Testing 

Cody

## Primary Program Loop / Stopping Test

Tanner

# Standards

- Use Numpy arrays instead of lists
- Write docstrings for your functions (googling "python docstrings" will give you an idea of how to do this)

## Git usage

- Develop in your own branch
- Submit pull requests when you're ready
- Anyone can approve a pull request, but don't approve your own
- Do a (very brief) code review on the PR before you approve it
- Feel free to log Issues if you feel the need
