import pandas as pd # pour afficher les matrices sous forme de tableau
import numpy as np 
import numpy.linalg as la
import numpy.polynomial.polynomial as nppol
import scipy as sp
import scipy.linalg as sla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importer Axes3D depuis la sous-bibliothèque mpl_toolkits.mplot3d
import scipy.optimize as sc
import scipy.integrate  as si # intégrale
import scipy.misc as derv
import random

#FONCTION 
def initialisation (taille, MaxX1, MinX1, MaxX2, MinX2):
    population = []
    for _ in range(taille):
        x = np.array([np.random.uniform(MinX1, MaxX1), np.random.uniform(MinX2, MaxX2)])
        obj = fonctionObj(x[0], x[1])             # Initialise fonction objective à 0
        individu = Individu(x, obj)
        population.append(individu)

    return population

def tournoi(Ntour, population):
    populationNew = [None] * len(population)
    for index, _ in enumerate(populationNew):
        max = Individu([0, 0], -np.inf)
        for i in np.random.choice(population, Ntour, replace=False):
            if max.obj < i.obj:
                max = i
        populationNew[index] = max  # Met à jour populationNew
    return populationNew

def Roue(Sp, population):
    tri = sorted(population, key=lambda individu: individu.obj)
    for individu in tri:
        print(individu)
    return population
    
#Affichage
def afficher_population(population):
    data = {
        'x1': [individu.x[0] for individu in population],
        'x2': [individu.x[1] for individu in population],
        'obj': [individu.obj for individu in population]
    }
    df = pd.DataFrame(data)
    print(df)
    
#Individu
class Individu:
    def __init__(self, x, obj):
        self.x = x
        self.obj = obj
    def __str__(self):
        return f'x1: {self.x[0]}, x2: {self.x[1]}, obj: {self.obj}'
###########################################################################################


#Fonction objective 
def fonctionObj(x1, x2) :
    return x1**2 + x2**2

#Initialisation
population = initialisation(100, -6, 6, -6, 6)
afficher_population(population)  # Affiche les valeurs x1, x2 et obj dans un DataFrame

#Selection
population = tournoi(4, population)
afficher_population(population)

Roue(4, population)
