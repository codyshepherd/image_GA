import pytest
from image_GA import *

TEST_IMAGE = "INPUT.PNG"

def testMeasureFitness():
    ind1 = Individual()
    pic1 = ind1.renderImage()
    assert(ind1.measureFitness(pic1) == 0)

    ind2 = Individual()
    assert(ind2.measureFitness(pic1) != 0)

def testMutate():
    ind1 = Individual()
    pic1 = ind1.renderImage()
    ind1.mutate(100)
    pic2 = ind1.renderImage()
    assert(np.sum(pic1-pic2) != 0)

def testCrossover():
    ind1 = Individual()
    ind2 = Individual()
    child1, child2 = Population.crossover(ind1, ind2)

    assert(np.sum(ind1.renderImage() - child1.renderImage()) != 0)
    assert(np.sum(ind1.renderImage() - child2.renderImage()) != 0)
    assert(np.sum(ind2.renderImage() - child1.renderImage()) != 0)
    assert(np.sum(ind2.renderImage() - child1.renderImage()) != 0)

def run_all():
    testMeasureFitness()
    testMutate()
    testCrossover()

run_all()
print("Huzzah")
