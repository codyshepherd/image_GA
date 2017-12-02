import numpy as np
import random
from PIL import Image, ImageDraw
import queue
import argparse
import uuid
import json

parser = argparse.ArgumentParser(description="Set Experiment Parameters")
parser.add_argument('-p', '--population', help='Population Size')
parser.add_argument('-pr', '--parent', help='Percentage the chosen parent is random, instead of the best')
parser.add_argument('-c', '--child', help='Child Mutation Probability [1-99]')
parser.add_argument('-sh', '--shape', help='Shape Mutation Probability [1-99]')
parser.add_argument('-hh', '--hard', help='Hard Mutation Probability [1-99]')
parser.add_argument('-m', '--medium', help='Medium Mutation Probability [1-99]')
parser.add_argument('-s', '--soft', help='Soft Mutation Probability [1-99]')
parser.add_argument('-d', '--delta', help='Mutation Delta [1-99]')
parser.add_argument('-t', '--tolerance', help='Stopping Tolerance [0.0-1.0]')
parser.add_argument('-q', '--queue', help='Max Queue Length')
parser.add_argument('-n', '--num_poly', help='Number of polygons per individual')
parser.add_argument('-i', '--iters', help='Minimum number of iterations to run')

nspace = vars(parser.parse_args())

i = nspace.get('iters')
MIN_STEPS =  int(i) if i else 100

# Performance tuning parameters
pop = nspace.get('population')
POPULATION_SIZE = int(pop) if pop else 20
n = nspace.get('num_poly')
NUM_POLYGONS = int(n) if n else 10
NUM_VERTICES = 3 # Start with triangle

# Dimensional maxes
X_MAX = 1024
Y_MAX = 1024
RGB_MAX = 255
ALPHA_MAX = 255

# Mutation Probabilites
pr = nspace.get('parent')
PERCENT_PARENT_RANDOM = int(pr) if pr else 10
c = nspace.get('child')
CHILD_MUTATION_PROB = int(c) if c else 30
sh = nspace.get('shape')
SHAPE_MUTATION_PROB = int(sh) if sh else 20
d = nspace.get('delta')
DELTA = int(d) if d else 20
COIN_TOSS = 2
h = nspace.get('hard')
HARD_MUTATION_PROB = int(h) if h else 10
m = nspace.get('medium')
MEDIUM_MUTATION_PROB = int(m) if m else 20
s = nspace.get('soft')
SOFT_MUTATION_PROB = int(s) if s else 30

if HARD_MUTATION_PROB + MEDIUM_MUTATION_PROB + SOFT_MUTATION_PROB >= 100:
    HARD_MUTATION_PROB = 10
    MEDIUM_MUTATION_PROB = 20
    SOFT_MUTATION_PROB = 30

#Stop conditions
t = nspace.get('tolerance')
TEST_STOP_TOLERANCE = float(t) if t else .1
q = nspace.get('queue')
MAX_FITNESS_QUEUE_LEN = int(q) if q else 20

#Input
IMAGE_FILE_PATH = "INPUT.PNG"
IMAGE = np.array([])
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
      populationArray = sorted(populationArray, key=lambda x: x.fitness)
      self.populationMembers = np.array(populationArray)

  @staticmethod
  def crossover(parentA,parentB):
    """
    Performs crossover of the two given parents of the population at a randomly 
    generated crosspoint and generates two children
    
    :param parentA: an individual
    :param parentB: an individual
    :return: a tuple of individuals
    """
    global IMAGE, IMAGE_FILE_PATH
    if IMAGE.size == 0:
        IMAGE = readOriginalImageFromFile(IMAGE_FILE_PATH)
   
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

  def getMaxFitnessIndividual(self):
    """Return the individual with the best fitness"""
    #return min(self.populationMembers, key=lambda x: x.fitness) # Lower fitness values are better since they represent a difference measure between images
    return self.populationMembers[0]

  def eliminateWeakest(self, child0, child1):
    """Finds the two weakest members of the population, including the children and kills them off"""
    newpop = sorted(np.concatenate((self.populationMembers, [child0, child1]), axis=0), key=lambda x: x.fitness)
    #newpop = sorted(list(np.concatenate(self.populationMembers, [child0, child1])), key=lambda x: x.fitness)
    self.populationMembers = np.array(newpop[:-2])

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
    global IMAGE, IMAGE_FILE_PATH
    if IMAGE.size == 0:
        IMAGE = readOriginalImageFromFile(IMAGE_FILE_PATH)
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
    """
    Measures the fitness via sum of squared difference of 
    pixel colors between orignal image and rendered solution. 

    :param originalImage: an nparray representing the comparison image
    :return: double representing the squared difference between the two arrays
    """
    if type(originalImage) != type(self.image):
      raise TypeError("images must be of same type!")

    return np.sum((self.image-originalImage)**2)

  def mutate(self):
    """
    Mutate one or more shapes within the individual.

    :return: None
    """
    counter = 0
    for shape in self.shapes:
      shape_mut = random.randrange(100)
      if shape_mut > SHAPE_MUTATION_PROB:
        continue
      spin_the_wheel = random.randrange(100)
      if spin_the_wheel <= HARD_MUTATION_PROB:
        shape.hard_mutate()
        counter += 1
        continue
      elif spin_the_wheel <= (MEDIUM_MUTATION_PROB + HARD_MUTATION_PROB):
        shape.medium_mutate()
        counter += 1
        continue
      elif spin_the_wheel <= (SOFT_MUTATION_PROB + MEDIUM_MUTATION_PROB + HARD_MUTATION_PROB):
        shape.soft_mutate()
        counter += 1
        
    #print("Individual mutated " + str(counter) + " shapes.")
    self.image = self.renderImage()
    self.fitness = self.measureFitness(IMAGE)

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

  def soft_mutate(self):
    """
    Mutate one paramater

    The coin toss will decide which parameter of the polygon
    will be changed. The delta will vary but overall should be a 
    minute change on the parameter being mutated

    :return none
    """
    #print("soft mutate")

    if random.randrange(COIN_TOSS) == 0:
      to_mutate = random.randrange(len(self.vertexList))
      index_to_change = random.randrange(len(self.vertexList[to_mutate]))
      change_param = self.vertexList[to_mutate][index_to_change]
      mutated_param = change_param + random.randint(-DELTA, DELTA)
      mutated_param = max(min(X_MAX, mutated_param), 0)
      self.vertexList[to_mutate][index_to_change] = mutated_param
    else:
      to_mutate = random.randrange(len(self.color))
      colors = list(self.color)
      color_change = colors[to_mutate]
      delta_color = color_change + random.randint(-DELTA, DELTA)
      mutated_color = max(min(RGB_MAX, delta_color), 0)
      colors[to_mutate] = mutated_color
      self.color = tuple(colors)

  def medium_mutate(self):
    """
    Mutate one paramater within the polygon

    Coin toss decides which paramter gets changed. This parameter will be
    changed to a random number within its min-max 

    :return none
    """
    #print("medium mutate")
    if random.randrange(COIN_TOSS) == 0:
      to_mutate = random.randrange(len(self.color))
      colors = list(self.color)
      mutated = random.randrange(RGB_MAX)
      colors[to_mutate] = mutated
      self.color = tuple(colors)
    else: 
      to_mutate = random.randrange(len(self.vertexList))
      change_coord = random.randrange(len(self.vertexList[to_mutate]))
      self.vertexList[to_mutate][change_coord] = random.randrange(X_MAX)
  
  def hard_mutate(self):
    """
    Mutate three paramters within the polygon

    Mutate a color, alpha, and a vertex. Each of the parameters
    mutated values will be random within its min-max 

    return: none
    """
    #print("hard mutate")
    colors = list(self.color)
    mutate_color = random.randrange(len(colors) - 1)
    colors[mutate_color] = random.randrange(RGB_MAX)
    colors[-1] = random.randrange(ALPHA_MAX)
    self.color = tuple(colors)

    mutate_vertex = random.randrange(len(self.vertexList))
    self.vertexList[mutate_vertex] = random.randrange(X_MAX), random.randrange(Y_MAX)

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
    if(oldFitness - maxFitness <= tolerance):
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
  """ Opens the image at specified <filePath>, converts to RGBA numpy array and returns it """
  image = Image.open(filePath)
  image = image.convert('RGBA')
  return np.array(image)

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
IMAGE = readOriginalImageFromFile(IMAGE_FILE_PATH)

imagePopulation = Population()

fitnessQueue = queue.Queue(MAX_FITNESS_QUEUE_LEN)

maxFitnessIndividual = imagePopulation.getMaxFitnessIndividual()
#evolutionComplete = evaluateStopCondition(fitnessQueue, MAX_FITNESS_QUEUE_LEN, TEST_STOP_TOLERANCE, maxFitnessIndividual.fitness)
evolutionComplete = False

counter = 0
while not evolutionComplete or counter < MIN_STEPS:
  rand_parent = random.randrange(100)
  if rand_parent <= PERCENT_PARENT_RANDOM:
    parentNum0 = random.randrange(POPULATION_SIZE)
    parentNum1 = random.randrange(POPULATION_SIZE)
    while(parentNum0 == parentNum1): # Avoid crossover with self
      parentNum1 = random.randrange(POPULATION_SIZE)

    parent0 = imagePopulation.populationMembers[parentNum0]
    parent1 = imagePopulation.populationMembers[parentNum1]
  else:
    parent0 = imagePopulation.populationMembers[0]
    parent1 = imagePopulation.populationMembers[1]

  child0, child1 = imagePopulation.crossover(parent0, parent1)
  child0.mutate()
  child1.mutate()
  imagePopulation.eliminateWeakest(child0,child1)
  maxFitnessIndividual = imagePopulation.getMaxFitnessIndividual()
  evolutionComplete = evaluateStopCondition(fitnessQueue, MAX_FITNESS_QUEUE_LEN, TEST_STOP_TOLERANCE, maxFitnessIndividual.fitness)
  counter += 1
  #print("Finished step " + str(counter))
  #print("Fitness queue: " + str(fitnessQueue.queue))
  #print("Pop fitnesses: " + str([x.fitness for x in imagePopulation.populationMembers]))
  #best = imagePopulation.getMaxFitnessIndividual()
  #best.saveImageToFile(str(counter))

print("Convergence")
best = imagePopulation.getMaxFitnessIndividual()
names = ['steps','pop','poly','shape','child','hard','med','soft','delt','tol','q','final']
params = [MIN_STEPS,POPULATION_SIZE,NUM_POLYGONS, SHAPE_MUTATION_PROB, CHILD_MUTATION_PROB, HARD_MUTATION_PROB, MEDIUM_MUTATION_PROB, SOFT_MUTATION_PROB, DELTA, TEST_STOP_TOLERANCE, MAX_FITNESS_QUEUE_LEN, best.fitness]
together = {}
for i in range(len(names)):
  together[names[i]] = str(params[i])
img_title = str(uuid.uuid4())
best.saveImageToFile(img_title)
with open(img_title + ".txt", 'w+') as fh:
  json.dump(together, fh)




