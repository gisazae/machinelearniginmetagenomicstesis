# importamos la librerias

import time
import numpy as np
import pandas as pd
# Funcion para vectorizar la secuencia.

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'A': '1000', 'C': '0100', 'G': '0010', 'T': '0001','N': '0000'}
    return [ltrdict[x] for x in seq]

origen_training_ctxm = open("training_nn_ctxm.txt", "r")#abrimos el archivo de entrenamiento
lines = origen_training_ctxm.readlines() #Leemos cada linea del archivo.
#salida_training_ctxm = open("sequences_nn_prueba.txt", "w") #escribimos el archivo de salida para el entrenamiento.
long_max = max([len(x.split(',')[0]) for x in lines] )

#print(long_max)

#Recorremos cada linea para dar formato al archivo de salida o de entranemiento
y_data = []
X_data = []
for line in lines:
    seq = line.split(",")[0]
    if (len(seq) < long_max):
        dif = long_max-len(seq)
        N = ''.join(['N' for x in range(dif)])
        seq += N

    out = float(line.split(",")[1])
    y_data.append(out)
    grupo_ctxm = int (line.split(",")[1])/100

    salida = vectorizeSequence(seq)

    vector = []
    for k in range(len(salida)):
        if salida[k] == '1000':
           vector.append(1)
           vector.append(0)
           vector.append(0)
           vector.append(0)
        if salida[k] == '0100':
            vector.append(0)
            vector.append(1)
            vector.append(0)
            vector.append(0)
        if salida[k] == '0010':
            vector.append(0)
            vector.append(0)
            vector.append(1)
            vector.append(0)
        if salida[k] == '0001':
            vector.append(0)
            vector.append(0)
            vector.append(0)
            vector.append(1)
        if salida[k] == '0000':
            vector.append(0)
            vector.append(0)
            vector.append(0)
            vector.append(0)

    X_data.append(vector)
    #salida_training_ctxm.write(''.join(salida) + "," + str(grupo_ctxm)+'\n')

X_data = np.array(X_data)
y_data = np.array(y_data)
y_data = np.expand_dims(y_data, axis=1)

print(X_data.shape)
print(y_data.shape)

colnames = []
for i in range(X_data.shape[1]):
    namelcol = str('nc')+str(i)
    colnames.append(namelcol)
colnames.append('y')

data = np.concatenate((X_data, y_data), axis=1)
print(data.shape)

df = pd.DataFrame(data=data, columns=colnames) 
print(df.head())
file_name = 'dataGen.csv'
#Cerramos los archivos abiertos y escribimos los tiempos de  procesamiento.
origen_training_ctxm.close()
df.to_csv(file_name, sep=',', encoding='utf-8')
print("Preparación del dataset lista.")
