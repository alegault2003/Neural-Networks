# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 07:33:14 2023

@author: 
"""
import matplotlib.pyplot as plt
plt.clf()
def plotWithLables(data, labels):
    # using some dummy data for this example
    xs = data[:, 1]
    ys = data[:, 0]
    labels = labels
    # 'bo-' means blue color, round points, solid lines
    plt.plot(xs,ys,'bo-')
    # zip joins x and y coordinates in pairs
    for x,y,l in zip(xs,ys, labels):
       
        plt.annotate(l,
                     (x,y), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center') 
    
    plt.show()
    
def plotNbChromosomesVsFitness(n_chromosomes, fitness):
    plt.plot(n_chromosomes, fitness)
    plt.title('Pop size vs fitness')
    plt.xlabel('pop size')
    plt.ylabel('fitness')
    