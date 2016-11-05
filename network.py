import random
import math
import copy
minli = -0.01
maxli = 0.01
alpha = 0.2

def sig(x):
  return 1 / (1 + math.exp(-x))

class Network:
    def __init__(self, layers):
        self.links = []
        self.layers = layers
        self.delta = copy.deepcopy(layers)
        for i in range(0, len(layers)-1):
            self.links.append([])
            for j in range(0, len(layers[i])):
                self.links[i].append([])
                for k in range(0, len(layers[i+1])):
                    self.links[i][j].append(random.uniform(minli, maxli))

    def __str__(self):
        s = ''
        for i in range(0, len(self.links)):
            s += 'li.' + str(i) + ':-> ' + str(self.links[i]) + '\n'
        for i in range(0, len(self.layers)):
            s += 'la.' + str(i) + ':-> ' + str(self.layers[i]) + '\n'
        return s[:-1] # remove last character (\n)

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

        # compute delta of output layer
        for i in range(0, len(self.layers[len(self.layers)-1])):
            self.delta[len(self.layers)-1][i] = expected[i] - self.layers[len(self.layers)-1][i]

        # compute delta of intern layers
        for i in reversed(range(1,len(self.layers)-1)):
            for j in range(0,len(self.layers[i])):
                t = 0
                for k in range(0, len(self.links[i][j])):
                    t += self.links[i][j][k] * self.delta[i+1][k]
                self.delta[i][j] = self.layers[i][j] * (1 - self.layers[i][j]) * t

        # update links with all deltas
        for i in range(0, len(self.links)):
            for j in range(0, len(self.links[i])):
                for k in range(0, len(self.links[i][j])):
                    self.links[i][j][k] = self.links[i][j][k] + alpha * self.layers[i][j] * self.delta[i+1][k]

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

def test():
    layers = [[0,0], [0,0], [0,0], [0]]
    n = Network(layers)
    n.setInputs([2,-1])
    n.setLinks(0, [[0.5,-1],[1.5,-2]])
    n.setLinks(1, [[1,-1],[3,-4]])
    n.setLinks(2, [[1],[-3]])
    print(str(n) + '\n')
    n.frontPropagation()
    print(str(n) + '\n')
    n.backPropagation([1])
    print(n)

# layers = [[0,0,0], [0,0,0,0], [0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0], [0,0,0], [0,0,0]] # +++
# layers = [[0,0,0], [0,0,0,0], [0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0], [0,0,0,0], [0,0,0]] # +
# layers = [[0,0,0], [0,0,0,0], [0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0], [0,0,0,0], [0,0,0]] # shit !
layers = [[0,0,0], [0,0,0,0], [0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0], [0,0,0]] # +++++ !
n = Network(layers)

print('Before training :')
print('Output for ' + str([0,0,0]) + ' : ' + str(n.run([0,0,0])))
print('Output for ' + str([1,0,0]) + ' : ' + str(n.run([1,0,0])))
print('Output for ' + str([0,0,1]) + ' : ' + str(n.run([0,0,1])))
print('Output for ' + str([0,1,0]) + ' : ' + str(n.run([0,1,0])))
print('Output for ' + str([1,0,1]) + ' : ' + str(n.run([1,0,1])))
print('Output for ' + str([1,1,0]) + ' : ' + str(n.run([1,1,0])))
print('Output for ' + str([0,1,1]) + ' : ' + str(n.run([0,1,1])))
print('Output for ' + str([1,1,1]) + ' : ' + str(n.run([1,1,1])))

print('\nTraining :')
trains = [[0,0,0],[1,0,0],[0,0,1],[0,1,0],[1,0,1],[1,1,0],[0,1,1],[1,1,1]]
results = [[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,0,1],[1,0,0],[0,0,0]]
for i in range(0,100001):
    if i % 10000 == 0:
        print('\t' + str(i/1000) + '%...')
    r = random.randint(0,len(trains)-1)
    n.train(trains[r], results[r])

print('\nAfter training :')
print('Output for ' + str([0,0,0]) + ' : ' + str(n.run([0,0,0])))
print('Output for ' + str([1,0,0]) + ' : ' + str(n.run([1,0,0])))
print('Output for ' + str([0,0,1]) + ' : ' + str(n.run([0,0,1])))
print('Output for ' + str([0,1,0]) + ' : ' + str(n.run([0,1,0])))
print('Output for ' + str([1,0,1]) + ' : ' + str(n.run([1,0,1])))
print('Output for ' + str([1,1,0]) + ' : ' + str(n.run([1,1,0])))
print('Output for ' + str([0,1,1]) + ' : ' + str(n.run([0,1,1])))
print('Output for ' + str([1,1,1]) + ' : ' + str(n.run([1,1,1])))
