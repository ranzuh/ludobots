import random
import numpy
import copy
import matplotlib.pyplot as plt

# FUNCTIONS

def MatrixCreate(rows, columns):
    matrix = numpy.zeros((rows, columns))
    return matrix
    
def MatrixRandomize(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = random.random()

def GetFitness(matrix):
    fitness = numpy.mean(matrix)
    return fitness
    
def MatrixPerturb(matrix, prob):
    matrix_copy =  copy.deepcopy(matrix)
    for i in range(len(matrix_copy)):
        for j in range(len(matrix_copy[i])):
            if prob > random.random():
                matrix_copy[i][j] = random.random()
    return matrix_copy

def PlotHillclimber(runs):
    plt.figure(figsize=(9,6))        
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    
    for i in range(runs):
        parent = MatrixCreate(1, 50)
        MatrixRandomize(parent)
        parentFitness = GetFitness(parent)
        fits = MatrixCreate(1, 5000)
        
        for currentGeneration in range(5000):
            child = MatrixPerturb(parent, 0.05) 
            childFitness = GetFitness(child)
            if childFitness > parentFitness:
                parent = child 
                parentFitness = childFitness
            fits[0][currentGeneration] = parentFitness
        
        plt.plot(fits[0])
    
    plt.show
    
def PlotGenes():
    genes = MatrixCreate(50, 5000)
    
    for gene in range(50):
        parent = MatrixCreate(1, 50)
        MatrixRandomize(parent)
        parentFitness = GetFitness(parent)
        for generation in range(5000):
            genes[gene][generation] = parent[0][gene]
            child = MatrixPerturb(parent, 0.05)
            childFitness = GetFitness(child)
            if childFitness > parentFitness:
                parent = child
                parentFitness = childFitness    
    
    plt.figure(figsize=(9,6))
    plt.xlabel("Generation")
    plt.ylabel("Gene")
    plt.imshow(genes, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.show()

# Run after uncommenting one of following lines

PlotHillclimber(5)
# PlotGenes()









