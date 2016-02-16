import random
import copy
import matplotlib.pyplot as plt
import numpy

def MatrixCreate(rows, columns):
    matrix = []
    for row in range(rows):
        matrix.append([0.0] * columns)
    return matrix

def MatrixRandomize(vector):
    for element, value in enumerate(vector[0]):
        vector[0][element] = random.random()

def Fitness(vector):
    ans = numpy.mean(vector[0])
    return ans

def MatrixPerturb(vector, prob):
    vectorCopy = copy.deepcopy(vector)
    for element, value in enumerate(vectorCopy[0]):
        if prob > random.random():
            vectorCopy[0][element] = random.random()
    return vectorCopy
    
#def PlotVectorAsLine(fits):
#    plt.plot(fits[0])
#    plt.show


genes = MatrixCreate(50, 5000)


for gene in range(50):
    parent = MatrixCreate(1, 50)
    MatrixRandomize(parent)
    parentFitness = Fitness(parent)
    for generation in range(5000):
        genes[gene][generation] = parent[0][gene]
        child = MatrixPerturb(parent, 0.05)
        childFitness = Fitness(child)
        if childFitness > parentFitness:
            parent = child
            parentFitness = childFitness
    
    
#for row in range(len(genes)):
#    parent = MatrixCreate(1, 50)
#    MatrixRandomize(parent)
#    parentFitness = Fitness(parent)
#    for currentGeneration in range(5000):
#        genes[row][currentGeneration] = parentFitness
#        child = MatrixPerturb(parent, 0.05)
#        childFitness = Fitness(child)
#        if childFitness > parentFitness:
#            parent = child
#            parentFitness = childFitness

plt.figure(figsize=(9,6))
plt.imshow(genes, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
plt.xlabel("Generation")
plt.ylabel("Gene")
plt.show()


# TESTING
# TESTING
# TESTING
# TESTING
# TESTING
        
