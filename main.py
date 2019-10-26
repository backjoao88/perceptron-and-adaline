from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math

class Perceptron:

  def __init__(self, x1, x2, x3, outputs, learningRate = 0.1   , epochs = 2000):
    
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3

    self.outputs = outputs
    self.weights = []
    self.bias = 0
    self.thresold = 0
    self.learningRate = learningRate
    self.epochs = epochs

  def initWeights(self):
    self.bias = 0
    for i in range(len(self.x1)):
      self.weights.append(0)
  #endfunction
  
  def recalcWeights(self, error, x, y, z):
    self.weights[0] = self.weights[0] + self.learningRate * error * x
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
    while(True):
      e = e + 1
      globalError = 0
      for i in range(len(self.x1)):
        result = self.calculateOutput(self.x1[i], self.x2[i], self.x3[i])
        print("Expected is {expected} and Predicted is {predicted}".format(expected=self.outputs[i], predicted=result))
        localError = self.outputs[i] - result
        self.recalcWeights(localError, self.x1[i], self.x2[i], self.x3[i])
        globalError += (localError * localError)
      print("Iteration => {e} with error {error}".format(e = e, error = math.sqrt(globalError/len(self.x1))))
      if(globalError == 0 or e >= self.epochs):
        break
    print("Decision boundary line => {weight1}*x1 + {weight2}*x2 + {weight3}*x3 + {bias} = 0".format(weight1=self.weights[0], weight2=self.weights[1], weight3=self.weights[2], bias=self.bias))
  #endfunction

def openFile(path):
  with open(path) as dataset:
    df = pd.read_excel(path)
    return df

dataset = openFile('Dados_Treinamento_Perceptron.xls')
p = Perceptron(dataset['x1'], dataset['x2'], dataset['x3'], dataset['d'])
p.train()