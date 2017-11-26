import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
import csv
style.use("ggplot")



with open("data.csv") as file:
    # esta data tiene 3333 filas (elementos de la lista results)
    data_param = csv.reader(file, delimiter=';')
    results = list(row for row in data_param)
    results.remove(results[0])
    results.remove(results[0])

# print(len(results[0][42]))

diccionario_variables = {"carac_{}".format(results.index(x)):
                             list(map(lambda num: float(num.replace(",",".")),x)) for x in results}


c=0
m = 0
total = 0
for i in diccionario_variables.keys():
    total += 1
    if diccionario_variables[i][0] == 1.0:
        c+=1
    else:
        m+=1
    #print(diccionario_variables[i])
print(c) # 1645
print(m) # 1689
print(c+m) # 3334
print(total)


# datos para el training, tomaremos 1600 por cada observación -1 y 1

data = [diccionario_variables[x] for x in diccionario_variables.keys()][0:1600]+\
       [diccionario_variables[x] for x in diccionario_variables.keys()][1689:3289]



W = np.array(data)
Z = [1 for j in range(int(len(data)/2))] + [-1 for i in range(int(len(data)/2))]

clf = svm.SVC(kernel = 'linear', C = 1.0)
# las caracteristicas estarian dadas por clf
clf.fit(W,Z)
# quien_quiero = np.array(quien_quiero).reshape((len(quien_quiero), -1))
# print(quien_quiero)
example = np.array(diccionario_variables["carac_1605"])
example = example.reshape(1,-1)
print(clf.predict(example))
w = clf.coef_[0]
print(w)

with open("my_result.txt", 'w', encoding="utf-8") as file:
    # esta data tiene 3333 filas (elementos de la lista results)
    file.write("                 Resultados Data grupo 7\n")
    file.write("---------------------------------------------------------------------\n")
    file.write("- Total de datos contenidos en la base:{}\n".format(total))
    file.write("- Training data usada: 3200 observaciones ([0:1600]->1,[1600:3200]->-1)\n")
    file.write("- Testing para la caracteristica 1605: 1\n")
    file.write("- Kernel: linear\n")
    file.write("- C keyargs for badly classify: 1.0")









# esto es solo para testear en con variables de dos diemnsiones
print()
print()




X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

Y = [-1,1,-1,1,-1,1]


clf = svm.SVC(kernel = 'linear', C = 1.0)
# las caracteristicas estarian dadas por clf
clf.fit(X,Y)
# quien_quiero = np.array(quien_quiero).reshape((len(quien_quiero), -1))
# print(quien_quiero)
example = np.array([0.58, 0.76])
example = example.reshape(1,-1)
print(clf.predict(example))
w = clf.coef_[0]
print(w)








