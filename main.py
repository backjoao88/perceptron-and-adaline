from pandas import DataFrame, read_csv
import pandas as pd
import random as random
import math

class Perceptron:
  def __init__(self, dataset, learningRate = 0.5, epochs = 1000):
    self.dataset = dataset
    self.weights = []
    self.outputs = []
    self.learningRate = learningRate
    self.epochs = epochs

  def initWeights(self):
    for i in range(len(self.dataset)):
      self.weights.append(random.random())

  def sigmoid(self, x):
    return (1 / (1 + math.exp(-x)))
  
  def recalcWeights(self, difference, inputs):
    for i in range(len(inputs)):
      self.weights[i] = self.weights[i] + self.learningRate * difference * inputs[i]
  def run(self, dataset):
    sum = 0
    for i in range(len(dataset)):
      sum = sum + (dataset)
  def train(self):
    self.initWeights()
    
    error = False
    i = 0

    while(error and i < self.epochs):
      difference = 0

      for i in range(len(self.dataset)):
        result = self.run(self.dataset[['x1', 'x2', 'x3']])
        if result != self.outputs[i]:
          error = True
          difference = self.outputs[i] - result
          self.recalcWeights(difference, p.dataset[['x1', 'x2', 'x3']])
        else:
          error = False
      print("Interaction => " + i)
      i = i + 1

def openFile(path):
    with open(path) as dataset:
      df = pd.read_excel(path)
      return df

p = Perceptron(openFile('Dados_Treinamento_Perceptron.xls'))
print(p.dataset[['x1', 'x2', 'x3']][0])



