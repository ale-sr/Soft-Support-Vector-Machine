# -*- coding: utf-8 -*-
"""

El siguiente [Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) consta de 7 grupos o clases con imágenes de expresiones faciales. 

Se pide:
 

*   Seleccionar las imágenes de  2 clases (angry, happy)
*   Separar aleatoriamente, 70% de training y el resto para testing.
*   Aplicar el algoritmo de [Haar Wavelet](https://pywavelets.readthedocs.io/en/latest/) para detectar caracetísticas más representativas del modelo.
*   Entrenar un soft SVM
*   Crear una matriz de confusión con los resultados obtenidos en el testing. 

[Tips](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)
"""

import numpy as np 
import math
import pandas as pd
import random
import matplotlib.pyplot as plt 
import cvxopt
import seaborn as sns
import itertools
import pywt
import os
from sklearn.metrics import confusion_matrix
import zipfile
from skimage import io, color

"""## Normalización"""

def normalizacion(data):
  min = np.min(data)
  max = np.max(data)
  if max == min:
    return np.ones(data.shape)
  return (data - min) / (max - min)

"""## Vector característico"""

# Cargar Base de Datos y seperar en grupo de entrenamiento y testing
with zipfile.ZipFile("/content/soft-svm.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/Images")

# Implente una función para tranformar cada imagen en un vector k dimensional
# mediante el haar wavelet
def haar(f, n):
  imagen = io.imread(f)
  #img = color.rgb2gray(data)
  for i in range(n):
    #coeffs2 = pywt.dwt2(imagen, 'haar')
    #imagen, (LH, HL, HH) = coeffs2
    imagen, (LH, HL, HH) = pywt.dwt2(imagen, 'haar')
  gfg = np.array(imagen)
  return gfg.flatten()

def encode(path):
  encodings = []
  for filename in os.listdir(path):
      f = os.path.join(path, filename)
      # checking if it is a file
      if os.path.isfile(f):
          encodings.append(haar(f, 3))
  return encodings

"""### Train"""

# Implemente la función de pérdida del soft SVM
def loss_function(x, y, w, c, b):
  return w**2/2 + c*sum([max(0, 1-y[i]*(np.dot(x[i], w.transpose()) + b)) for i in range(y.size)])

# Implemente la función para obtener las derivadas de W
def derivadas(x, y, w, b, c):
  mask = np.heaviside(1 - y * (np.dot(x, w.transpose()) + b), 0)
  dw = w + c*-np.dot(y * mask, x)
  db = c * -np.dot(y, mask)
  return db, dw

# Implemente la función para cambiar los W
def change_parameters(w, b, db, dw, alpha):
  # for j in range(k):
  #   w[j] = w[j] - alpha*dw[j]
  w = w - alpha*dw
  b = b - alpha*db
  return w, b

# Implemente la función de training. 
def training(x, y, c, alpha, epochs, xt = np.array([]), yt = np.array([])):
  w = np.array([np.random.rand() for i in range(k)])
  b = np.random.rand()
  errtr = []
  errts = []
  cont = 0
  while cont < epochs:
    db, dw = derivadas(x, y, w, b, c)
    w, b = change_parameters(w, b, db, dw, alpha)
    L = loss_function(x, y, w, c, b)
    errtr.append(L)
    if xt.any() and yt.any():
      errts.append(loss_function(xt, yt, w, c, b))
    cont += 1
  return w, b, errtr, errts

# Implemente la función de testing
def testing(X,W,b):
  Y_result = []
  # write your code here
  for i in range(m):
    f_xj = (np.dot(X[i], W.transpose())+b)
    if f_xj >= 0:
      Y_result.append(1)
    else:
      Y_result.append(-1)
  return np.array(Y_result)

def compare(y1, y2):
  return sum([y1[i]==y2[i] for i in range(len(y1))])

"""### Main"""

def get_data(happy, angry):
  data  =  np.array(encode(happy))
  data  = np.insert(data, 0, 1, axis=1)

  temp =  np.array(encode(angry))
  temp = np.insert(temp, 0, -1, axis=1)

  data = np.concatenate((data, temp), axis=0)

  for i in range(10):
    np.random.shuffle(data)

  y = data[:,0]
  x = data[:, 1:]

  return x, y

happy_path_train  = '/content/Images/images/images/train/happy'
angry_path_train  = '/content/Images/images/images/train/angry'

happy_path_test  = '/content/Images/images/images/validation/happy'
angry_path_test  = '/content/Images/images/images/validation/angry'

train_x, train_y = get_data(happy_path_train, angry_path_train)
test_x, test_y = get_data(happy_path_test, angry_path_test)

n = train_y.size
k = train_x[0].size

train_x_norm = np.apply_along_axis(normalizacion, 1, train_x)
test_x_norm = np.apply_along_axis(normalizacion, 1, test_x)

W, b, e1, e2 = training(train_x_norm, train_y, 1e6, 1e-8, 1200, test_x_norm, test_y)

m = test_y.size
y_pred = testing(test_x_norm, W, b)

test_y = test_y.astype('int')

correct = compare(test_y, y_pred)
print("Clasificados correctamente:", correct)
print("Clasificados incorrectamente:", len(test_y)-correct)
print("% de efectividad", round(100*correct/len(test_y), 2))

matrix = confusion_matrix(test_y,y_pred)
df2 = pd.DataFrame(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], index=["Angry", 'Happy'], columns=["Angry", 'Happy'])
sns.heatmap(df2, annot=True, cbar=None, cmap="Greens")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.show()