import csv
import os
from gurobipy import *


with open("data.csv") as file:
    # esta data tiene 3333 filas (elementos de la lista results)
    data_param = csv.reader(file, delimiter=';')
    results = list(row for row in data_param)
    results.remove(results[0])
    results.remove(results[0])

#print(len(results[0][42]))

diccionario_variables = {"carac_{}".format(results.index(x)): x for x in results}

#for i in diccionario_variables.keys():
#    print(i)

class SVM:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colores = {1:'r', -1:'b'}

