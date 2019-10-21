from pandas import DataFrame, read_csv
import pandas as pd
import random as random

class Perceptron:
  def __init__(self, dataset, learningRate = 0.5, epochs = 1000):
    self.dataset = dataset
    self.weights = []
    self.outputs = []
    self.learningRate = learningRate
    self.epochs = epochs

  def train(self):
    self.outputs = self.dataset[4]
    # Initializing array of wecights
    print(self.outputs)
    for i in range(self.dataset):
      self.weights.append(random.random())

def openFile(path):
    with open(path) as dataset:
      df = pd.read_excel(path)
      print(df)

p = Perceptron(openFile('Dados_Treinamento_Perceptron.xls'))
p.train()



