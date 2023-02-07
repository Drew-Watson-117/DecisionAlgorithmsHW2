import helperFunctions as hf
from multiprocessing import Pool
import numpy as np

# This function can find the most optimal epsilon by convergence speed or average reward
def mostOptimalEpsilon(iterations, epsilons=[0.01, 0.05, 0.1, 0.4], drift=False, fastest = True):
    mostOptimalEpsilon = 0
    optimalArray = [0] * len(epsilons)
    averageConvergenceSpeed = [0] * len(epsilons)
    averageReward = [0]*len(epsilons)
    if fastest:
        arguments = [(epsilons,drift) for i in range(iterations)]
        with Pool() as pool:
            for epsilon, speed in pool.starmap(findFastestEpsilon, arguments):
                index = epsilons.index(epsilon)
                optimalArray[index] += 1
                averageConvergenceSpeed[index] += speed 
            pool.close()
        index = np.argmax(optimalArray) # Most common optimum epsilon
        mostOptimalEpsilon = epsilons[index]
        convergenceSpeed = averageConvergenceSpeed[index] / optimalArray[index] # Have to divide by how many times chosen
        print("Optimal Epsilon for Convergence Speed")
        print(f"Trials: {iterations}")
        print(f"Optimal Epsilon: {mostOptimalEpsilon}")
        print(f"Average Convergence Speed: {convergenceSpeed} iterations\n")
    else:
        arguments = [(epsilons,drift) for i in range(iterations)]
        with Pool() as pool:
            for epsilon, speed in pool.starmap(findRichestEpsilon, arguments):
                index = epsilons.index(epsilon)
                optimalArray[index] += 1
                averageReward[index] += speed 
            pool.close()
        index = np.argmax(optimalArray) # Most common optimum epsilon
        mostOptimalEpsilon = epsilons[index]
        reward = averageReward[index] / optimalArray[index] # Have to divide by how many times chosen
        print("Optimal Epsilon for Average Reward")
        print(f"Trials: {iterations}")
        print(f"Optimal Epsilon: {mostOptimalEpsilon}")
        print(f"Average Reward: {reward}\n")

def findFastestEpsilon(epsilons, drift=False):
    optimalEpsilon = -1.
    minSpeed = 1000000000000
    for epsilon in epsilons:
        speed = computeEpsilonConvergenceSpeed(epsilon, drift=drift)
        if speed < minSpeed:
            minSpeed = speed
            optimalEpsilon = epsilon
    return optimalEpsilon, minSpeed

def findRichestEpsilon(epsilons, drift=False):
    optimalEpsilon = -1.
    bestReward = -1000000000000
    for epsilon in epsilons:
        averageReward = epsilonGreedy(epsilon,drift=drift, graph=False, output=False)
        if averageReward > bestReward:
            bestReward = averageReward
            optimalEpsilon = epsilon
    return optimalEpsilon, bestReward

def computeEpsilonConvergenceSpeed(epsilon, iterations=1000, drift=False):
    probabilities = hf.get_probabilities()
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    convergencePercentage = 0.5
    returnValue = 50*iterations
    for t in range(1,iterations):
        if drift:
            probabilities = hf.get_probabilities(-0.001,t)
        else:
            probabilities = hf.get_probabilities()

        # If the most picked is more than convergencePercentage of the iterations so far
        if max(N) / t > convergencePercentage and (np.argmax(N)==2 or np.argmax(N)==17) and t > 20:
            returnValue = t
            break
        else:
            epsilonGreedyStep(epsilon,probabilities,Q,N)
    return returnValue

def epsilonGreedy(epsilon, iterations=10000, graph=True, drift=False, output=True):
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
    if graph:
        hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for epsilon={epsilon}")
    if output:
        print(f"Epsilon Greedy:\n Epsilon = {epsilon}\n Average Reward: {averageReward}\n Most Chosen Arm: {np.argmax(N)}")
    else:
        return averageReward
    

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