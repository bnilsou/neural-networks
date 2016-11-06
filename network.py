import random
import math
import copy
import ast

def sig(x):
  return 1 / (1 + math.exp(-x))

class Network:
    def __init__(self, layers, alpha, mi, ma):
        self.links = []
        self.layers = []
        self.deltas = []
        self.alpha = alpha
        self.min = mi
        self.max = ma
        for i in layers:
            self.layers.append([0] * i)
            self.deltas.append([0] * i)

        for i in range(0, len(self.layers)-1):
            self.links.append([])
            for j in range(0, len(self.layers[i])):
                self.links[i].append([])
                for k in range(0, len(self.layers[i+1])):
                    self.links[i][j].append(random.uniform(self.min, self.max))

    def __str__(self):
        s = ''
        for i in range(0, len(self.links)):
            s += 'li.' + str(i) + ':-> ' + str(self.links[i]) + '\n'
        for i in range(0, len(self.layers)):
            s += 'la.' + str(i) + ':-> ' + str(self.layers[i]) + '\n'
        return s[:-1] # remove last character (\n)

    def save(self, name):
        layers = []
        for i in range(0, len(self.layers)):
            layers.append(len(self.layers[i]))
        file = open(name + '.nnw', 'w')
        file.write(str(layers) + '\n')
        file.write(str(self.links) + '\n')
        file.close()

    def load(self, name):
        file = open(name, 'r')
        layers = ast.literal_eval(file.readline())
        self.links = ast.literal_eval(file.readline())
        file.close()
        self.layers = []
        self.deltas = []
        for i in layers:
            self.layers.append([0] * i)
            self.deltas.append([0] * i)

    def setInputs(self, inputs):
        if len(inputs) != len(self.layers[0]):
            raise ValueError('Length of inputs is invalid')
        self.layers[0] = inputs

    def setLinks(self, l, links):
        self.links[l] = links

    def frontPropagation(self):
        for i in range(1,len(self.layers)):
            for j in range(0,len(self.layers[i])):
                n = 0
                for k in range(0,len(self.layers[i-1])):
                    n += self.layers[i-1][k] * self.links[i-1][k][j]
                self.layers[i][j] = sig(n)

    def backPropagation(self, expected):
        if len(expected) != len(self.layers[len(self.layers)-1]):
            raise ValueError('Length of expected is invalid')

        # compute deltas of output layer
        for i in range(0, len(self.layers[len(self.layers)-1])):
            self.deltas[len(self.layers)-1][i] = expected[i] - self.layers[len(self.layers)-1][i]

        # compute deltas of intern layers
        for i in reversed(range(1,len(self.layers)-1)):
            for j in range(0,len(self.layers[i])):
                t = 0
                for k in range(0, len(self.links[i][j])):
                    t += self.links[i][j][k] * self.deltas[i+1][k]
                self.deltas[i][j] = self.layers[i][j] * (1 - self.layers[i][j]) * t

        # update links with all deltas
        for i in range(0, len(self.links)):
            for j in range(0, len(self.links[i])):
                for k in range(0, len(self.links[i][j])):
                    self.links[i][j][k] = self.links[i][j][k] + self.alpha * self.layers[i][j] * self.deltas[i+1][k]

    def train(self, inputs, expected):
        self.setInputs(inputs)
        self.frontPropagation()
        self.backPropagation(expected)

    def run(self, inputs):
        self.setInputs(inputs)
        for i in range(1, len(self.layers)):
            for j in range(0, len(self.layers[i])):
                n = 0
                for k in range(0,len(self.layers[i-1])):
                    n += self.layers[i-1][k] * self.links[i-1][k][j]
                self.layers[i][j] = sig(n) # What ?
        for i in range(0, len(self.layers[len(self.layers)-1])):
            self.layers[len(self.layers)-1][i] = round(self.layers[len(self.layers)-1][i], 3)
        return self.layers[len(self.layers)-1]
