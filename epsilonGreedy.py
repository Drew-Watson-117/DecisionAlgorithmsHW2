import helperFunctions as hf
from multiprocessing import Pool
import numpy as np

def averageOptimalEpsilon(iterations, subdivisions):
    averageOptimalEpsilon = 0
    arguments = [subdivisions for i in range(iterations)]
    with Pool() as pool:
        for result in pool.imap(findOptimalEpsilon, arguments):
            averageOptimalEpsilon += result
    averageOptimalEpsilon /= iterations
    convergenceSpeed = computeEpsilonConvergenceSpeed(averageOptimalEpsilon)
    print(f"The average optimal epsilon after {iterations} with {subdivisions} steps was found to be {averageOptimalEpsilon}. The convergence speed for is value was {convergenceSpeed}")
        
def findOptimalEpsilon(subdivisions=10):
    optimalEpsilon = 0
    minSpeed = 1000000000000
    for i in range(1,subdivisions):
        epsilon = i / subdivisions
        speed = computeEpsilonConvergenceSpeed(epsilon)
        if speed < minSpeed:
            minSpeed = speed
            optimalEpsilon = epsilon
    return optimalEpsilon

def computeEpsilonConvergenceSpeed(epsilon, iterations=1000):
    probabilities = hf.get_probabilities()
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    convergencePercentage = 0.8
    returnValue = 50*iterations
    for t in range(1,iterations):
        probabilities = hf.get_probabilities()
        # If the most picked is more than convergencePercentage of the iterations so far
        if max(N) / t > convergencePercentage and (np.argmax(N)==2 or np.argmax(N)==17):
            returnValue = t
            break
        else:
            epsilonGreedyStep(epsilon,probabilities,Q,N)
    return returnValue

def epsilonGreedy(epsilon, iterations=10000):
    probabilities = hf.get_probabilities()

    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t

    tArray = np.arange(0,iterations)
    nMatrix = np.matrix([[0]*iterations]*len(N))
    for t in range(iterations):
        probabilities = hf.get_probabilities()
        epsilonGreedyStep(epsilon,probabilities,Q,N)
        for i in range(len(N)):
            nMatrix.itemset((i,t),N[i])
    print(f"Epsilon Greedy:\nEpsilon = {epsilon}, Expected Values for Arms:\n {Q}")
    hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for epsilon={epsilon}")
    

def epsilonGreedyStep(epsilon,probabilities,Q,N):
    if np.random.random() < epsilon:
        # Explore
        index = np.random.randint(0,len(probabilities))
    else:
        # Exploit
        index = np.argmax(Q)

    result = probabilities[index]
    # Update chosen action
    Q[index] = hf.updateExpected(result, Q[index], N[index])
    N[index] += 1 # Might need to be after updateExpected -- Then get a /0 error
