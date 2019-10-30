from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math

class Adaline:

  def __init__(self, x1, x2, x3, outputs, learningRate = 0.0001, requiredPrecision = 0.1, epochs = 2000):
    
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3

    self.outputs = outputs
    self.weights = []
    self.bias = 0
    self.learningRate = learningRate
    self.requiredPrecision = requiredPrecision
    self.epochs = epochs

  def EQM(self):
    sizeTrainingSample = len(self.x1)
    EQM = 0
    for i in range(len(sizeTrainingSample)):
      u = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
      EQM += EQM + ((self.outputs[i] - u) ** 2)
    EQM = EQM / sizeTrainingSample
    return EQM
  #endfunction

  def initWeights(self):
    self.bias = 0
    for i in range(len(self.x1)):
      self.weights.append(0)
  #endfunction
  
  def recalcWeights(self, error, x, y, z):
    # EQM
    self.weights[0] = self.weights[0] * learningRate
    self.weights[1] = self.weights[1] + self.learningRate * error * y
    self.weights[2] = self.weights[2] + self.learningRate * error * z
    self.bias = self.bias + self.learningRate * error
  #endfunction
  
  def calculateOutput(self, x, y, z):
    sum = x * self.weights[0] + y * self.weights[1] +z * self.weights[2] + self.bias
    if(sum >= self.thresold):
      return 1
    else:
      return -1
  #endfunction

  def train(self):
    self.initWeights()
    e = 0

    previousEQM = self.EQM()
    while(True):
      for i in range(len(self.x1)):
        result = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
        self.recalcWeights(localError, self.x1[i], self.x2[i], self.x3[i])
      e = e + 1
      currentEQM = self.EQM()
      print("Current EQM -> {eqm}".format(eqm = currentEQM))
      if (currentEQM - previousEQM) < self.requiredPrecision:
        break
    return [self.weights[0], self.weights[1], self.weights[2], self.bias]
  #endfunction

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df
#endfunction

dataset = openFile('Dados_Treinamento_Perceptron.xls')

p = Adaline(dataset['x1'], dataset['x2'], dataset['x3'], dataset['d'])
p.train()



