import numpy as np
import random
import copy
import matplotlib.pyplot as plt


def MatrixCreate(rows, columns):
    matrix = np.zeros((rows, columns))
    return matrix

def MatrixRandomize(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = random.uniform(-1,1)

def MatrixPerturb(matrix, prob):
    matrix_copy =  copy.deepcopy(matrix)
    for i in range(len(matrix_copy)):
        for j in range(len(matrix_copy[i])):
            if prob > random.random():
                matrix_copy[i][j] = random.uniform(-1,1)
    return matrix_copy

def Fitness(neuronValues):
    actualNeuronValues = neuronValues[9,:]
    
    desiredNeuronValues = MatrixCreate(1,10)
    for j in range(1,10,2):
        desiredNeuronValues[0][j] = 1

    dist = MeanDistance(actualNeuronValues, desiredNeuronValues)

    fitness = 1 - dist

    return fitness

def Fitness2(neuronValues):
    diff=0.0

    for i in range(1,9): 

          for j in range(0,9):

               diff=diff + abs(neuronValues[i,j]-neuronValues[i,j+1])

               diff=diff + abs(neuronValues[i+1,j]-neuronValues[i,j]) 

    diff=diff/(2*8*9)

    return diff

def UpdateNeurons(neuronValues, synapses, row):
    for j in range(10):
        tempSum = 0

        for k in range(10):
            tempSum += neuronValues[row-1,k] * synapses[j,k]

        if tempSum < 0: tempSum = 0
        elif tempSum > 1: tempSum = 1

        neuronValues[row,j] = tempSum

def MeanDistance(v1, v2):
    d = ((v1 - v2) ** 2).mean()
    return d

def NeuronValues(synapses):
    neuronValues = MatrixCreate(10,10)
    
    for j in range(len(neuronValues[0])):
        neuronValues[0][j] = 0.5

    for i in range(1,10):
        UpdateNeurons(neuronValues, synapses, i)

    return neuronValues

def ShowMatrixImage(neuronValues):
    plt.figure(figsize=(9,6))
    plt.xlabel("Neuron")
    plt.ylabel("Time step")
    plt.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.show()

parentSynapses = MatrixCreate(50,50) 
MatrixRandomize(parentSynapses)

parentNeurons = NeuronValues(parentSynapses)
ShowMatrixImage(parentNeurons)

parentFitness = Fitness2(parentNeurons)

fitness = MatrixCreate(1,1000)

for currentGeneration in range(1000):

    fitness[0][currentGeneration] = parentFitness

    childSynapses = MatrixPerturb(parentSynapses, 0.05)

    childNeurons = NeuronValues(childSynapses)

    childFitness = Fitness2(childNeurons)

    if (childFitness > parentFitness):
        parentSynapses = childSynapses
        parentFitness = childFitness

parentNeurons = NeuronValues(parentSynapses)
ShowMatrixImage(parentNeurons)

plt.figure(figsize=(9,6))
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.plot(fitness[0])
plt.show()







