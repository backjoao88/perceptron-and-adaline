from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df

def printOnFile(line):
  with open("tests/PerceptronTests.txt", "a+", encoding="utf-8") as file:
    file.write(line + "\n")

def writeList(lst):
  for item in lst:
    printOnFile(item) 

class Perceptron:

  def __init__(self, x1, x2, x3, outputs, learningRate = 0.1   , epochs = 100):
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.outputs = outputs
    self.weights = []
    self.bias = 0
    self.learningRate = learningRate
    self.epochs = epochs
    self.trainingCounter = 0

  def initWeights(self):
    self.bias = random.random()
    for i in range(len(self.x1)):
      self.weights.append(random.random())
  #endfunction
  
  def recalcWeights(self, error, x, y, z):
    self.weights[0] = self.weights[0] + self.learningRate * error * x
    self.weights[1] = self.weights[1] + self.learningRate * error * y
    self.weights[2] = self.weights[2] + self.learningRate * error * z
    self.bias = self.bias + self.learningRate * error
  #endfunction
  
  def calculateOutput(self, x, y, z):
    sum = x * self.weights[0] + y * self.weights[1] +z * self.weights[2] + self.bias
    if(sum >= 0):
      return 1
    else:
      return -1
  #endfunction

  def validate(self, dataset):
    line = self.train()
    self.predicted = []

    printOnFile("-> Validation Data")
    printOnFile("X1 = [" + ", ".join(str(v) for v in datasetV['x1']) + "]")
    printOnFile("x2 = [" + ", ".join(str(v) for v in datasetV['x2']) + "]")
    printOnFile("X3 = [" + ", ".join(str(v) for v in datasetV['x3']) + "]")

    for i in range(len(datasetV['x1'])):
      sum = datasetV['x1'][i] * line[0] + datasetV['x2'][i] * line[1] + datasetV['x3'][i] * line[2] + line[3]
      if(sum >= 0):
        result = 1
      else:
        result = -1
      self.predicted.append(result)
    
    printOnFile("-> Predicted Outputs")
    printOnFile("[" + ", ".join(str(v) for v in self.predicted) + "]")

  #endfunction

  def train(self):
    self.initWeights()
    self.trainingCounter += 1
    printOnFile("-> Test Number %d " % (self.trainingCounter))
    printOnFile("Initial Weights -> [%f, %f, %f, %f] " % (self.weights[0], self.weights[1], self.weights[2], self.bias))
    printOnFile("Learning Rate -> %f" % (self.learningRate))
    e = 0
    while(True):
      e = e + 1
      globalError = 0
      for i in range(len(self.x1)):
        result = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
        localError = self.outputs[i] - result
        self.recalcWeights(localError, self.x1[i], self.x2[i], self.x3[i])
        globalError += (localError * localError)
      if(globalError == 0 or e >= self.epochs):
        break
    printOnFile("Final Weights -> [%f, %f, %f, %f] " % (self.weights[0], self.weights[1], self.weights[2], self.bias))
    printOnFile("Epochs -> %d " % (e))
    printOnFile("Global Error -> %f " % (globalError))

    return [self.weights[0], self.weights[1], self.weights[2], self.bias]
  #endfunction

dataset = openFile('datasets/Dados_Treinamento_Perceptron.xls')
datasetV = openFile('datasets/Dados_Validação_Perceptron.xls')

p = Perceptron(dataset['x1'], dataset['x2'], dataset['x3'], dataset['d'])

printOnFile("=========================================================================")
p.validate(datasetV)
printOnFile("=========================================================================")
