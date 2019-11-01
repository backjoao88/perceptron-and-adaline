from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math


class Adaline:

  def __init__(self, x1, x2, x3, outputs, learningRate = 0.01, requiredPrecision = 0.0001):
    
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
    return [self.weights[0], self.weights[1], self.weights[2], self.bias]
  #endfunction

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df
#endfunction

dataset = openFile('Dados_Treinamento_Perceptron.xls')
datasetV = openFile('Dados_Validação_Perceptron.xls')

p = Adaline(dataset['x1'], dataset['x2'], dataset['x3'], dataset['d'])
#p.validate(dataset['x1'], dataset['x2'], dataset['x3'])
p.train()



# => Treinamento: 
# Pesos Sinápticos - Weights (Obs: Posição 0 é o bias):
# Iniciais => [0. 0. 0. 0.]
# Finais   => [2.6260084416312894, 1.5496181751568099, 2.3748149335864213, -0.6750827626238209]
# Número de Épocas => 405
# Taxa de Aprendizado => 0.01
# Precisão requerida => 1e-05