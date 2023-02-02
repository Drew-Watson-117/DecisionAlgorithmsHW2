import helperFunctions as hf
import numpy as np
import matplotlib.pyplot as plt

def main():
    # epsilonGreedy(0.01)
    # epsilonGreedy(0.05)
    # epsilonGreedy(0.1)
    # epsilonGreedy(0.4)
    # thompsonSample()
    optimalEpsilon()

def optimalEpsilon():
    iterations = 500
    minSpeed = 1000000000000
    optimalEpsilon = 0
    for i in range(0,iterations):
        epsilon = i / iterations
        speed, arm = computeConvergenceSpeed(epsilon)
        if speed < minSpeed:
            minSpeed = speed
            optimalEpsilon = epsilon
    print(f"The optimal epsilon for this problem is {optimalEpsilon}, which has a convergence speed of {minSpeed} iterations to Arm {arm}")

def computeConvergenceSpeed(epsilon):
    probabilities = hf.get_probabilities()
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    iterations = 1000
    convergencePercentage = 0.8
    for t in range(1,iterations):
        probabilities = hf.get_probabilities()
        # If the most picked is more than convergencePercentage of the iterations so far
        if t > 6 and max(N) / t > convergencePercentage and (np.argmax(N)==2 or np.argmax(N)==17):
            return t, np.argmax(N)
        else:
            epsilonGreedyStep(epsilon,probabilities,Q,N)
    return iterations, np.argmax(N)


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

    hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for epsilon={epsilon}")
    print(f"Epsilon = {epsilon}, Expected Values for Arms:\n {Q}")

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


def thompsonSample(iterations=10000):
    probabilities = hf.get_probabilities()


    Q = [np.random.beta(1.,1.)] * len(probabilities) # List of EXPECTED probabilities
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    alpha = [1.] * iterations
    beta = [1.] * iterations

    tArray = np.arange(0,iterations)
    nMatrix = np.matrix([[0]*iterations]*len(N))
    
    for t in range(iterations):
        probabilities = hf.get_probabilities()
        thompsonSampleStep(Q, N, probabilities, alpha, beta)
        for i in range(len(N)):
            nMatrix.itemset((i,t),N[i])
    hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for Thompson Sampling")

def thompsonSampleStep(Q,N,probabilities,alpha,beta):
    index = np.argmax(Q)
    result = probabilities[index]
    if result > 0:
        alpha[index] += result
    else:
        beta[index] -= result
    Q[index] = np.random.beta(alpha[index],beta[index])
    N[index]+= 1





 
if __name__=="__main__":
    main()
