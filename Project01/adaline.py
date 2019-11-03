from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math


class Adaline:

  def __init__(self, x1, x2, x3, outputs, learningRate = 0.01, requiredPrecision = 0.00001):
    
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.outputs = outputs
    
    self.weights = []
    self.bias = 0
    self.learningRate = learningRate
    self.requiredPrecision = requiredPrecision

  def EQM(self):
    sizeTrainingSample = len(self.x1)
    EQM = 0
    #print(self.x1)
    for i in range(len(self.x1)):
      u = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
      EQM += (self.outputs[i] - u) ** 2
    EQM = EQM / sizeTrainingSample
    return EQM
  #endfunction

  def initWeights(self):
    self.bias = 0
    for i in range(len(self.x1)):
      self.weights.append(0)
  #endfunction
  
  def calculateOutput(self, x, y, z):
    output = x * self.weights[0] + y * self.weights[1] + z * self.weights[2] + self.bias
    return output
  #endfunction

  def validate(self, x1, x2, x3):
    line = self.train()
    self.predicted = []

    printOnFile("-> Validation Data")
    printOnFile("X1 = [" + ", ".join(str(v) for v in x1) + "]")
    printOnFile("X2 = [" + ", ".join(str(v) for v in x2) + "]")
    printOnFile("X3 = [" + ", ".join(str(v) for v in x3) + "]")

    for i in range(len(x1)):
      sum = x1[i] * line[0] + x2[i] * line[1] + x3[i] * line[2] + line[3]
      if(sum >= 0):
        result = 1
      else:
        result = -1
      self.predicted.append(result)
    
    printOnFile("-> Predicted Outputs")
    printOnFile("[" + ", ".join(str(v) for v in self.predicted) + "]")


  def train(self):
    self.initWeights()

    printOnFile("-> Test Number %d " % (0))
    printOnFile("Initial Weights -> [%f, %f, %f, %f] " % (self.weights[0], self.weights[1], self.weights[2], self.bias))
    printOnFile("Learning Rate -> %f" % (self.learningRate))
    printOnFile("Required Precision -> %f" % (self.requiredPrecision))

    e = 0
    currentEQM  = float("inf")
    previousEQM = 0
    globalError = 0
    while(True):
      previousEQM = self.EQM()
      for i in range(len(self.x1)):
        u = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
        self.weights[0] += self.learningRate * (self.outputs[i] - u) * self.x1[i]
        self.weights[1] += self.learningRate * (self.outputs[i] - u) * self.x2[i]
        self.weights[2] += self.learningRate * (self.outputs[i] - u) * self.x3[i]
        self.bias       += self.learningRate * (self.outputs[i] - u) 
      e = e + 1
      currentEQM = self.EQM()
      print("{weight1}*x1 + {weight2}*x2 + {weight3}*x3 + {bias} = 0".format(weight1=self.weights[0], weight2=self.weights[1], weight3=self.weights[2], bias=self.bias))
      
      if abs(currentEQM - previousEQM) <= self.requiredPrecision:
        break

    printOnFile("Final Weights -> [%f, %f, %f, %f] " % (self.weights[0], self.weights[1], self.weights[2], self.bias))
    printOnFile("Epochs -> %d " % (e))
    printOnFile("Final EQM -> %f " % (abs(currentEQM - previousEQM)))

    return [self.weights[0], self.weights[1], self.weights[2], self.bias]
  #endfunction

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df
#endfunction

def printOnFile(line):
  with open("tests/AdalineTests.txt", "a+", encoding="utf-8") as file:
    file.write(line + "\n")

def writeList(lst):
  for item in lst:
    printOnFile(item) 


dataset = openFile('datasets/Dados_Treinamento_Perceptron.xls')
datasetV = openFile('datasets/Dados_Validação_Perceptron.xls')

p = Adaline(dataset['x1'], dataset['x2'], dataset['x3'], dataset['d'])
p.validate(dataset['x1'], dataset['x2'], dataset['x3'])
#p.train()

