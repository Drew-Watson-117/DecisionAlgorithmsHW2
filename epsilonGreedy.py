import helperFunctions as hf
from multiprocessing import Pool
import numpy as np

def averageOptimalEpsilon(iterations, subdivisions, drift=False):
    averageOptimalEpsilon = 0

    arguments = [(subdivisions,drift) for i in range(iterations)]
    with Pool() as pool:
        for result in pool.starmap(findOptimalEpsilon, arguments):
            averageOptimalEpsilon += result
        pool.close()

    averageOptimalEpsilon /= iterations
    convergenceSpeed = computeEpsilonConvergenceSpeed(averageOptimalEpsilon, drift=drift)
    print(f"The average optimal epsilon after {iterations} with {subdivisions} steps was found to be {averageOptimalEpsilon}.\n The convergence speed for this value was {convergenceSpeed}")
        
def findOptimalEpsilon(subdivisions=10, drift=False):
    optimalEpsilon = -1.
    minSpeed = 1000000000000
    for i in range(1,subdivisions):
        epsilon = i / subdivisions
        speed = computeEpsilonConvergenceSpeed(epsilon, drift=drift)
        if speed < minSpeed:
            minSpeed = speed
            optimalEpsilon = epsilon
    return optimalEpsilon

def computeEpsilonConvergenceSpeed(epsilon, iterations=1000, drift=False):
    probabilities = hf.get_probabilities()
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    convergencePercentage = 0.8
    returnValue = 50*iterations
    for t in range(1,iterations):
        if drift:
            probabilities = hf.get_probabilities(-0.001,t)
        else:
            probabilities = hf.get_probabilities()

        # If the most picked is more than convergencePercentage of the iterations so far
        if max(N) / t > convergencePercentage and (np.argmax(N)==2 or np.argmax(N)==17) and t > 10:
            returnValue = t
            break
        else:
            epsilonGreedyStep(epsilon,probabilities,Q,N)
    return returnValue

def epsilonGreedy(epsilon, iterations=10000, graph=True, drift=False):
    probabilities = hf.get_probabilities()

    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    averageReward = 0

    if graph:
        tArray = np.arange(0,iterations)
        nMatrix = np.matrix([[0]*iterations]*len(N))
    for t in range(iterations):
        if drift:
            probabilities = hf.get_probabilities(-0.001,t)
        else:
            probabilities = hf.get_probabilities()
        averageReward += epsilonGreedyStep(epsilon,probabilities,Q,N, True)
        if graph:
            for i in range(len(N)):
                nMatrix.itemset((i,t),N[i])
    averageReward /= iterations
    print(f"Epsilon Greedy:\n Epsilon = {epsilon}\n Average Reward: {averageReward}\n Most Chosen Arm: {np.argmax(N)}")
    if graph:
        hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for epsilon={epsilon}")
    

def epsilonGreedyStep(epsilon,probabilities,Q,N, averageReward=False):
    if np.random.random() < epsilon:
        # Explore
        index = np.random.randint(0,len(probabilities))
    else:
        # Exploit
        index = np.argmax(Q)

    result = probabilities[index]
    averageReward += result
    # Update chosen action
    Q[index] = hf.updateExpected(result, Q[index], N[index])
    N[index] += 1 # Might need to be after updateExpected -- Then get a /0 error
    if averageReward:
        return result