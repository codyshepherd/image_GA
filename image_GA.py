import numpy as np
import random
import queue

# Performance tuning parameters
POPULATION_SIZE = 20
NUM_POLYGONS = 50
NUM_VERTICES = 3 # Start with triangle

# Dimensional maxes
X_MAX = 1024
Y_MAX = 1024
RGB_MAX = 255
ALPHA_MAX = 1

# Mutation Probabilites
CHILD_MUTATION_PROB = 30
SHAPE_MUTATION_PROB = 10

#Stop conditions
TEST_STOP_TOLERANCE = .1 
MAX_FITNESS_QUEUE_LEN = 100
MAX_ITERATIONS = 100000

#Input
IMAGE = None
RANDOM_SEED = 1
random.seed = RANDOM_SEED

#############################################################
# Class Definitions
#
#############################################################

class Population:
  """Holds a list with all members of the population."""

  def __init__(self, populationMembers = None):
    """Class Constructor

    Allows population to be initialized to a list of individuals. If none are
    provided a random group of individuals will be generated. 
    """
    if(populationMembers):
      self.populationMembers = populationMembers
    else:
      populationArray = []
      for i in range(POPULATION_SIZE):
        populationArray.append(Individual())
      self.populationMembers = np.array(populationArray)

  def crossover(self):
    """Randomly selects two members of the population and generates two children via crossover."""
    pass


class Individual:
  """Defines a set of polygons which form an image.

  This class holds the "DNA" for an individual. Its data represents a possible
  solution to the rendering problem. An individual's solution is formed of a 
  list of polygons which form an image when overlayed. An individual should
  also store a rendered version of their data as a bitmap as well as the fitness 
  of their solution.
  """

  def __init__(self, shapes = None):
    """Class Constructor

    Allows individual to be initialized to a list of polygons that are passed in.
    If no list is passed in the individual will random generate an intial solution.
    """
    if(shapes):
      self.shapes = shapes
    else:
      shapeArray = []
      for i in range(NUM_POLYGONS):
      	shapeArray.append(Shape())
      self.shapes = np.array(shapeArray)

    self.image = self.renderImage()
    self.fitness = self.measureFitness(IMAGE)

  def renderImage(self):
    """Renders a bitmaps from a list of shapes that should be overlayed"""
    pass

  def measureFitness(self, originalImage):
    """Measures the fitness via sum of squared difference of pixel colors between orignal image and rendered solution. """
    pass

  def mutate(self):
    """Mutate one or more shapes within the individual."""
    pass


class Shape:
  """Defines a single polygon. 

  This class specifies the vertices and color for a single polygon. Vertices
  are represented with (x,y) tuples and colors are represented with (r,g,b,a)
  tuples. 
  """

  def __init__(self, vertexList = None, color = None):
    """Class Constructor

    Optionally takes in list of vertices and a color for a polgyon. 
    If these are not provided a random set of vertices and a color will
    be chosen randomly
    """ 
    if(vertexList):
      self.vertexList = vertexList
    else:
      vertices = []
      for i in range(NUM_VERTICES):
        vertices.append(self.randomVertex())
      self.vertexList = np.array(vertices, dtype = ('int, int'))

    if(color):
      self.color = color
    else:
      self.color = self.randomColor()

  def randomVertex(self):
    """ Generate a random vertex tuple"""
    return (random.randrange(X_MAX), random.randrange(Y_MAX))

  def randomColor(self):
  	""" Generate a random color tuple"""
  	return (random.randrange(RGB_MAX), random.randrange(RGB_MAX), random.randrange(RGB_MAX), random.random())

  def mutate(self):
  	"""Mutates one or more vertex or the color of the polgyon."""
  	pass

  def print(self):
    """Print some debug information in an easy to read format"""
    print("vertexList = " + str(self.vertexList))
    print("color = " + str(self.color))


#############################################################
# Global Functions 
#
#############################################################
def evaluateStopCondition(fitnessQueue, maxLength, tolerance, maxFitness):
  """Checks if the algorithm should stop.should

  Pushes the current maximum fitness on a queue and pops an older one off. 
  When the difference between the current and old fitness is smaller than a
  set tolerance the algorithm will stop. 
  """
  if(fitnessQueue.qsize() == maxLength): # Only pop value if queue is full
    oldFitness = fitnessQueue.get()
    if(maxFitness - oldFitness <= tolerance):
      return True
  fitnessQueue.put(maxFitness)
  return False

def printFitnessQueue(fitnessQueue):
  """Debug function. Print a list of some of the recent fitness scores. """
  i = 0
  for fitness in fitnessQueue.queue:
    print("Fitness " + str(i) + ": " + str(fitness))
    i += 1

#############################################################
# Unit Tests 
#
#############################################################
def classInstantiationTest():
  """Instantiate all classes and print to prove they exist"""
  polygon = Shape()
  polygon.print()
  individual = Individual()
  for shape in individual.shapes:
    shape.print()
  population = Population()
  for individual in population.populationMembers:
    for shape in individual.shapes:
      shape.print()

def stopTest():
  """Instantiate a fitnessQueue, fill it up, and test stop conditions.

  Test will push growing fitnesses to the queue over time but the growth
  will level off over time. When it passes some threshold the test should 
  stop and print the queue. 
  """
  fitnessQueue = queue.Queue(MAX_FITNESS_QUEUE_LEN) # Create a queue
  i = 0
  j = 0
  stop = False
  while(not stop):
    stop = evaluateStopCondition(fitnessQueue, MAX_FITNESS_QUEUE_LEN, 1, i)
    i += 5-j # Adding higher fitness on each iteration but level off over time
    if(j < 5): # Dont ever subtract performance over an iteration
      j += .125
  printFitnessQueue(fitnessQueue)

#############################################################
# Main Loop
#
#############################################################

