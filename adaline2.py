from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math


class Adaline:

  def __init__(self, x1, x2, x3, x4, outputs, learningRate = 0.01, requiredPrecision = 0.00001):
    
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.x4 = x4
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
      u = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i], self.x4[i])
      EQM += (self.outputs[i] - u) ** 2
    EQM = EQM / sizeTrainingSample
    return EQM
  #endfunction

  def initWeights(self):
    self.bias = 0
    for i in range(len(self.x1)):
      self.weights.append(0)
  #endfunction
  
  def calculateOutput(self, x, y, z, p):
    output = x * self.weights[0] + y * self.weights[1] + z * self.weights[2] + p * self.weights[3] + self.bias
    return output
  #endfunction

  def validate(self, x1, x2, x3):
    line = self.train()
    for i in range(len(x1)):
      sum = x1[i] * line[0] + x2[i] * line[1] + x3[i] * line[2] + line[3]
      if(sum >= 0):
        result = 1
      else:
        result = -1
      print("Iteration = {i}".format(i=i))
      print("x1={x1}, x2={x2}, x3={x3} = {result}".format(x1=x1[i], x2=x2[i], x3=x3[i], result=result))


  def train(self):
    self.initWeights()
    e = 0
    currentEQM  = float("inf")
    previousEQM = 0
    globalError = 0
    while(True):
      previousEQM = self.EQM()
      for i in range(len(self.x1)):
        u = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i], self.x4[i])
        self.weights[0] += self.learningRate * (self.outputs[i] - u) * self.x1[i]
        self.weights[1] += self.learningRate * (self.outputs[i] - u) * self.x2[i]
        self.weights[2] += self.learningRate * (self.outputs[i] - u) * self.x3[i]
        self.weights[3] += self.learningRate * (self.outputs[i] - u) * self.x4[i]
        self.bias       += self.learningRate * (self.outputs[i] - u) 
      e = e + 1
      currentEQM = self.EQM()
      print("{weight1}*x1 + {weight2}*x2 + {weight3}*x3 + {weight4}*x4 + {bias} = 0".format(weight1=self.weights[0], weight2=self.weights[1], weight3=self.weights[2], weight4=self.weights[3], bias=self.bias))
      if abs(currentEQM - previousEQM) <= self.requiredPrecision:
        break
    return [self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.bias]
  #endfunction

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df
#endfunction

dataset = openFile('Dados_Treinamento_Adaline.xls')
print(dataset['x1'])
p = Adaline(dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], dataset['d'])
#p.validate(dataset['x1'], dataset['x2'], dataset['x3'])
p.train()

