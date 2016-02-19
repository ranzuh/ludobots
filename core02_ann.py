import random
import numpy
import copy
import matplotlib.pyplot as plt
from math import pi,sin,cos

# FUNCTIONS

def MatrixCreate(rows, columns):
    matrix = numpy.zeros((rows, columns))
    return matrix
    
def MatrixRandomize(matrix, rows):
    for i in range(rows):
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


def GetNeurons():
    neuronValues = MatrixCreate(50,10)
    MatrixRandomize(neuronValues, 1)
    
    return neuronValues

def GetNeuronPos():
    neuronPositions = MatrixCreate(2,10)
    
    numNeurons = 10
    angle = 0.0
    angleUpdate = 2 * pi / numNeurons
    
    for i in range(numNeurons):    
        neuronPositions[0,i] = sin(angle)    
        neuronPositions[1,i] = cos(angle)
        angle += angleUpdate
    
    return neuronPositions

def GetSynapses():
    synapses = MatrixCreate(10,10)

    for i in range(len(synapses)):
        for j in range(len(synapses[i])):
            synapses[i][j] = random.uniform(-1,1)
    
    return synapses

def PlotNeurons(neuronPositions):
    
    plt.plot(neuronPositions[0], neuronPositions[1],
             "ko", color=[1,1,1], markersize=18)

def PlotSynapses(neuronPositions, synapses):    
    x = neuronPositions[0]
    y = neuronPositions[1]
    
    
    for i in range(10):
        for j in range(10):
            w = int(10 * abs(synapses[i,j])) + 1
            if(synapses[i,j] < 0):
                plt.plot([x[i],x[j]], [y[i],y[j]], color=[0.8,0.8,0.8], linewidth=w)
            else:
                plt.plot([x[i],x[j]], [y[i],y[j]], color=[0,0,0], linewidth=w)

def UpdateNeurons(neuronValues, synapses, row):
    for j in range(10):
        tempSum = 0
        
        for k in range(10):
            tempSum += neuronValues[row-1,k] * synapses[j,k]
        
        if tempSum < 0: tempSum = 0
        elif tempSum > 1: tempSum = 1
        
        neuronValues[row,j] = tempSum



neuronValues = GetNeurons()
synapses = GetSynapses()

for i in range(1, 50):
    UpdateNeurons(neuronValues, synapses, i)

plt.figure(figsize=(9,6))
plt.xlabel("Neuron")
plt.ylabel("Time step")
plt.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')




#plt.figure(figsize=(9,6))
#PlotSynapses(neurons, synapses)
#PlotNeurons(neurons)























