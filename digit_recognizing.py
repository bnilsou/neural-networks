import gzip, pickle
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import network as nt

global train_x, train_y

def importData(data = 'mnist.pkl.gz'):
    global train_x, train_y
    print('Reading data...')
    with gzip.open(data, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set
    print('\tDone !\n')

def show(n):
    global train_x, train_y
    print('You are reading \'' + str(train_y[n]) + '\'')
    plt.imshow(train_x[n].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()

##################
# Import data
########
importData()

print('Initializing network...')
network = nt.Network([28*28,10], 0.2, -0.01, 0.01)
print('\tDone !\n')

##################
# Training network
########
# print('Training network...')
# for i in range(0, len(train_x)):
#     if (i+1) % 500 == 0:
#         print('\t' + str((i+1)/500) + '%...')
#     expected = [0] * 10
#     expected[train_y[i]] = 1
#     network.train(train_x[i], expected)
#
# print('Saving network...')
# network.save('digits')
# print('\tDone !\n')

print('Loading network...')
network.load('digits.nnw')
print('\tDone !\n')

##################
# Testing network with data
########
print('Testing...')
successes = 0
failures = 0
for i in range(0,len(train_x)-1):
    if (i+1) % 500 == 0:
        print('\t' + str((i+1)/500) + '%...')
    r = network.run(train_x[i])

    best = 0
    resMax = 0
    index = 0
    for k in r:
        if k > resMax:
            best = index
            resMax = k
        index += 1

    if best == train_y[i]:
        successes += 1
    else:
        failures += 1
print('\tSuccesses : ' + str(successes) + ' (' + str(successes/500) + '%)')
print('\tFailures : ' + str(failures) + ' (' + str(failures/500) + '%)')
print('\tDone !\n')
