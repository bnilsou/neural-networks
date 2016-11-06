import random
import network as nt

def test1():
    layers = [2,2,2,1]
    n = nt.Network(layers, 0.2, -0.01, 0.01)
    n.setInputs([2,-1])
    n.setLinks(0, [[0.5,-1],[1.5,-2]])
    n.setLinks(1, [[1,-1],[3,-4]])
    n.setLinks(2, [[1],[-3]])
    print(str(n) + '\n')
    n.frontPropagation()
    print(str(n) + '\n')
    n.backPropagation([1])
    print(n)

def test2():
    # layers = [3,4,5,6,5,3,3] # +++
    # layers = [3,4,5,6,7,5,4,3] # +
    # layers = [3,4,5,6,6,6,5,4,3] # shit !
    layers = [3,4,5,6,5,3] # +++++ !
    n = nt.Network(layers, 0.2, -0.01, 0.01)

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
    n.save('first')

def test3():
    n = nt.Network([0,0], 0.2, -0.01, 0.01)
    n.load('first.nnw')
    print('Output for ' + str([0,0,0]) + ' : ' + str(n.run([0,0,0])))
    print('Output for ' + str([1,0,0]) + ' : ' + str(n.run([1,0,0])))
    print('Output for ' + str([0,0,1]) + ' : ' + str(n.run([0,0,1])))
    print('Output for ' + str([0,1,0]) + ' : ' + str(n.run([0,1,0])))
    print('Output for ' + str([1,0,1]) + ' : ' + str(n.run([1,0,1])))
    print('Output for ' + str([1,1,0]) + ' : ' + str(n.run([1,1,0])))
    print('Output for ' + str([0,1,1]) + ' : ' + str(n.run([0,1,1])))
    print('Output for ' + str([1,1,1]) + ' : ' + str(n.run([1,1,1])))

print('Test 1:')
test1()
print('\nTest 2:')
test2()
print('\nTest 3:')
test3()
