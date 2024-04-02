#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
"""
import random
import numpy as np
import myPlot as mp
import pandas as pd

def getCitiesNames(bestPath, cities):
    bestPathNames = []
    for i in range(len(bestPath)-1):
        coord = bestPath[i]
        x, y = coord[0], coord[1]
        for j in range(len(cities)):
            lat = cities['lat'][j] 
            lng = cities['lng'][j]
            if x == lat and y == lng:
                bestPathNames.append(cities['city'][j])
    return bestPathNames

# generate a new population of size n
def generatePopulation(n, cities_coordinates):
    chromosomes = []
    for i in range(n):
        chromosomes.append(random.sample(cities_coordinates, len(cities_coordinates)))
    return chromosomes

# Manhattan distance
def manhattan(city1, city2):
    return abs(city1[0] - city2[0]) + abs(city1[1] - city2[1])

# fitness calculation for one chromosome
def findFitness(chromosome):
    
    fitness = 0.0
    
    # Manhattan distance between every 2 citie (from first until last)
    for i in range(len(chromosome)-1):
        fitness += manhattan(chromosome[i], chromosome[i+1])
        
    # Manhattan distance between the last and first city
    fitness += manhattan(chromosome[len(chromosome)-1], chromosome[0])
    
    return fitness
   
# calculate the cumulative probability
def getCumulative(fitnessList):
    
    # find inverse of fitness since we want to get the minimum number (shortest path)
    inverseFitness = []
    for f in fitnessList:
        inverseFitness.append(1/f)
        
    totalFitness = sum(inverseFitness)

    probabilityCount = []
    
    for i in inverseFitness:
        probabilityCount.append(i / totalFitness)
    
    cumulative = []
    current = 0
    for p in probabilityCount:
        current += p
        cumulative.append(current)

    return probabilityCount, cumulative

# print fitness table
def displayFitness(generation):
    
    fitness = []
    
    for chromosome in generation:
        fitness.append(findFitness(chromosome))
        
    probabilityCount, cumulative = getCumulative(fitness)

    # print fitness table
    print("\n %-6s%-11s%-16s%-13s%-10s" % ("no.", "fitness", "inverse", "probability", "cumulative"))
    print(" %-17s%-16s%-13s%-10s" % ("", "fitness", "count", "probability"))
    print("-"*60)
    for i in range(len(fitness)):
        print(" %-6d%-11.1f%-16.9f%-13.6f%-12.6f" 
              % (i+1, fitness[i], 1/fitness[i], probabilityCount[i], cumulative[i]))

# finds the best fitness and path with best fitness in a population
def findBestFitness(population):
    
    bestFitness = findFitness(population[0])
    bestPath = population[0]
    
    for path in population:
        fitness = findFitness(path)
        if fitness < bestFitness:
            bestFitness = fitness
            bestPath = path
           
    
    return bestFitness, bestPath

# choose parents(chromosomes) for mating using Roulette Wheel
def chooseParents(chromosomes, cumulative):
    
    parents = []
    probabilities = []
    
    for i in range(len(chromosomes)):
        probabilities.append(random.random())
    
    for i in range(len(chromosomes)):
        for curr in range(len(chromosomes)):
            if probabilities[i] < cumulative[curr]:
                parents.append(chromosomes[curr])
                break
            
    return parents

#  choose pairs for crossover
def choosePairs(x):
    
    # split the chromosome into 2 equal parts
    halfSize = int(len(x)/2)

    a = x[0:halfSize:]
    b = x[halfSize:len(x):]
    
    pairs = []
    for i in range(len(a)):
        y = [a[i], b[i]]
        pairs.append(y)

        # to make sure none of the pairs have the same chromosomes
        for p in pairs:
            if p[0] == p[1]:
                
                for j in range(len(pairs)):
                    
                    if pairs[j][0] != p[0] and pairs[j][1] != p[0]:
                        temp = p[0]
                        p[0] = pairs[j][0]
                        pairs[j][0] = temp
                        break
                    else:
                        continue
            
    return pairs

# One Point Crossover, a crossover point on the parent string is selected. 
# All data after that point in the list is swapped between the two parents.
def crossover(parent1, parent2):
    
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    #to make sure there is no repeatiton 
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    
    return child
  
# mutation by taking 2 random cities in a path (chromosome) and swapping them
def mutation(A):
    
    # get 2 random different indexes
    indx1 = random.randint(0, len(A)-1)
    indx2 = random.randint(0, len(A)-1)
    while indx2 == indx1:
        indx2 = random.randint(0, len(A)-1)
    
    temp = A[indx1]
    A[indx1] = A[indx2]
    A[indx2] = temp
    
    return A

# genetic algorithm
def GA(chromosomes, crossoverProbability, mutationProbability):

    # calculate the fitness and cumulative probability
    fitnessList = []
    for chromosome in chromosomes:
        fitnessList.append(findFitness(chromosome))
        
    cumulative = getCumulative(fitnessList)[1]
    
    # choose the parents for mating using Roulette Wheel
    parents = chooseParents(chromosomes, cumulative)
    
    # choose the pairs for crossover
    pairs = choosePairs(parents)

    children = []
    
    # crossover
    for p in pairs:
        r = random.random()
        # if crossover happenes
        if r < crossoverProbability:
            child1 = crossover(p[0], p[1])
            child2 = crossover(p[1], p[0])
            
        # if crossover does not happen
        else:
            child1, child2 = p[0], p[1]
            
        # appened the chromosomes to the list of children
        children.append(child1)
        children.append(child2)
        
    newGeneration = []
    
    # mutation
    for c in children:
        r = random.random()
        # if mutation happens
        if r < mutationProbability:
            child = mutation(c)
        # if mutation does not happen
        else:
            child = c
        newGeneration.append(child)
    
    return newGeneration

## /////////////////////////////////   Start from here //////////////////////////////////////

# load them each coordinate in a list
# cities_coordinates is the list of coordinates

cities = pd.read_csv('canada_cities.csv')
cities_name = cities['city'].values
coord = cities[['lat', 'lng']].values
cities_coordinates = [list(x) for x in coord]

# number of chromosomes
n_chromosomes = 60

# generate the population
chromosomes = generatePopulation(n_chromosomes, cities_coordinates)

# display fitness of original population
print("Number of chromosomes: ", n_chromosomes, "\n")
print("Fitness of original population:\n")
displayFitness(chromosomes)

# display best fitness
bestFitness, bestPath = findBestFitness(chromosomes)
print("\n Best fitness = ", bestFitness, "\n ")
print("-"*60)

# crossover and mutation with different probabilities
crossoverV = 0.5 
mutationV = 0.2 

bestGeneration = chromosomes.copy()     
newGeneration = GA(chromosomes, crossoverV, mutationV)

fitness = findBestFitness(newGeneration)[0]
   
if fitness < bestFitness:
    bestGeneration = newGeneration.copy()
    bestFitness = fitness
    
chromosomes = newGeneration.copy()
        
# calculate the fitness and cumulative probability of the best new generation
print("\n\nFitness of the best new generation:\n")
displayFitness(bestGeneration)

# display the best path in each different population size
bestFitness, bestPath = findBestFitness(bestGeneration)

print("\n Best fitness = ", bestFitness, "\n ")
print("-"*60)


# stop at the start point
bestPath.append(bestPath[0])

bestPathNames = getCitiesNames(bestPath, cities)     
print('Best path with cities names', bestPathNames)

# plot the path with cities labels
data = np.array(bestPath)
mp.plotWithLables(data, bestPathNames)

# plot fitness vs number chromosomes
#mp.plotNbChromosomesVsFitness(n_chromosomes, best_fitness_list)         
            
            
            
            
            
            
            
            
            
            
            
            
            
            