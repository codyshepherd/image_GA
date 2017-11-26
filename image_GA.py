import numpy as np
import random
from PIL import Image, ImageDraw
import queue


# Performance tuning parameters
POPULATION_SIZE = 20
NUM_POLYGONS = 50
NUM_VERTICES = 3 # Start with triangle

# Dimensional maxes
X_MAX = 1024
Y_MAX = 1024
RGB_MAX = 255
ALPHA_MAX = 255

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

  def crossover(self,parentA,parentB):
    """Performs crossover of the two given parents of the population at a randomly generated crosspoint
       and generates two children"""
   
    offspringA = Individual()
    offspringB = Individual()

    offspringA_temp_list = []
    offspringB_temp_list = []
   
    cross_over_point = random.randrange(NUM_POLYGONS)
  
    # print ("Crossover point " + str(cross_over_point))

    # Concatenates two parent arrays after splitting at the crossover point into a list
    offspringA_temp_list =  np.concatenate((parentA.shapes[:cross_over_point],parentB.shapes[cross_over_point:]))
    offspringB_temp_list =  np.concatenate((parentB.shapes[:cross_over_point],parentA.shapes[cross_over_point:]))
    
    offspringA.shapes = np.array(offspringA_temp_list)
    offspringB.shapes = np.array(offspringB_temp_list)
                                  
    # Fitness is recalculated for new offspring
    offspringA.image = offspringA.renderImage()
    offspringA.fitness = offspringA.measureFitness(IMAGE)

    offspringB.image = offspringB.renderImage()
    offspringB.fitness = offspringB.measureFitness(IMAGE)
    
    return offspringA,offspringB


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
    """Creates an image from a list of shapes that should be overlayed and returns np array of pixels"""

    # Creates white background base image, needed to preserve transparency (can be of any color)
    finalImage = Image.new('RGBA', (X_MAX, Y_MAX), color=(RGB_MAX, RGB_MAX, RGB_MAX, ALPHA_MAX))

    for shape in self.shapes:
      layer= Image.new('RGBA', (X_MAX, Y_MAX))
      ImageDraw.Draw(layer).polygon(shape.vertexList.tolist(), fill = shape.color) # tolist() provides required comma-separated input to polygon function
      finalImage = Image.alpha_composite(finalImage, layer)

    return np.array(finalImage)

  def saveImageToFile(self, fileName):
    """ Creates image and saves it to file '<fileName>.png' """
    image = Image.fromarray(self.image, 'RGBA')
    image.save(fileName + ".png")

  def measureFitness(self, originalImage):
    """Measures the fitness via sum of squared difference of pixel colors between orignal image and rendered solution. """
    if originalImage == None:
      raise TypeError("originalImage is None, must be numpy array")
    return np.sum((self.image-originalImage)**2)

  def mutate(self):
    """Mutate one or more shapes within the individual."""
    for shape in self.shapes:
      if random.randrange(100) == SHAPE_MUTATION_PROB:
        shape.mutate()

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
    if(vertexList is not None): # if(vertexList) is not used because np array cannot be checked as boolean
      self.vertexList =  vertexList
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
  	return (random.randrange(RGB_MAX), random.randrange(RGB_MAX), random.randrange(RGB_MAX), random.randrange(ALPHA_MAX))

  def mutate(self):
  	"""Mutates one or more vertex or the color of the polgyon."""
    self.vertexList[random.randrange(NUM_VERTICES)] = self.randomVertex()
    self.color = self.randomColor()

    # Give chance for another vertex to be changed 
    if random.randrange(100) ==  SHAPE_MUTATION_PROB:
      self.vertexList[random.randrange(NUM_VERTICES)] = self.randomVertex()

  def print(self):
    """Print some debug information in an easy to read format"""
    print("vertexList = " + str(self.vertexList))
    print("color = " + str(self.color))


#############################################################
# Global Functions 
#
#############################################################
def evaluateStopCondition(fitnessQueue, maxLength, tolerance, maxFitness):
  """Checks if the algorithm should stop.

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

def readOriginalImageFromFile(filePath):
  """ Opens the image at specified <filePath>, converts to RGBA numpy array and saves to IMAGE """
  image = Image.open(filePath)
  image = image.convert('RGBA')
  global IMAGE
  IMAGE = np.array(image)

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

def crossoverTest():
  """Creates a population and calls the crossover method on two randomly generated parents
     Prints the shapes in both parent and  offspring individual"""
  population = Population()

  # Randomly generates two parents for time-being  
  parentA = Individual()
  parentB = Individual()

  parentA.shapes = population.populationMembers[random.randrange(POPULATION_SIZE)].shapes
  parentB.shapes = population.populationMembers[random.randrange(POPULATION_SIZE)].shapes

  print("PARENT A")
  for shape in parentA.shapes:
    shape.print()
  print("PARENT B")
  for shape in parentB.shapes:
    shape.print()

  print("Crossing over")
  childA,childB = population.crossover(parentA,parentB)

  print("CHILD A")
  for shape in childA.shapes:
    shape.print()
  print("CHILD B")
  for shape in childB.shapes:
    shape.print()

def imageRenderingTest():
  """Renders a Star formed of 5 overlapping triangle shapes"""

  # Need to read original image first of all to be able to measure fitness
  readOriginalImageFromFile("INPUT.PNG")  

  # Triangle1
  point_1 = (300, 200)
  point_2 = (500, 800)
  point_3 = (700, 200)
  vertexList_1 = np.array([point_1, point_2, point_3], dtype = ('int, int'))

  # Triangle2
  point_4 = (700, 200)
  point_5 = (200, 600)
  point_6 = (800, 600)
  vertexList_2 = np.array([point_4, point_5, point_6], dtype = ('int, int'))

  # Triangle3
  point_7 = (800, 600)
  point_8 = (300, 200)
  point_9 = (500, 800)
  vertexList_3 = np.array([point_7, point_8, point_9], dtype = ('int, int'))

  # Triangle4
  point_10 = (500, 800)
  point_11 = (700, 200)
  point_12 = (200, 600)
  vertexList_4 = np.array([point_10, point_11, point_12], dtype = ('int, int'))

  # Triangle5
  point_13 = (200, 600)
  point_14 = (800, 600)
  point_15 = (300, 200)
  vertexList_5 = np.array([point_13, point_14, point_15], dtype = ('int, int'))

  # Instantiates shape object for all triangles
  ## Order of the co-ordinates matter in a way that first and last
  ## coordinates should be connected as one of polygon boundries
  shape_1 = Shape(vertexList_1, (250,10,10,50))
  shape_2 = Shape(vertexList_2, (10,250,10,50))
  shape_3 = Shape(vertexList_3, (10,10,250,50))
  shape_4 = Shape(vertexList_4, (250,10,10,50))
  shape_5 = Shape(vertexList_5, (10,250,10,50))

  # Instantiates individual that in turn will call renderImage
  individual = Individual([shape_1, shape_2, shape_3, shape_4, shape_5])

  # Prints fitness and saves output image to file 
  print("Fitness achieved = " + str(individual.fitness))
  individual.saveImageToFile("OUTPUT")


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
