import helperFunctions as hf
import numpy as np


def thompsonSample(iterations=10000, graph=True, drift=False, reset=False):
    probabilities = hf.get_probabilities()


    Q = [np.random.beta(1.,1.) for i in range(len(probabilities))] # List of EXPECTED probabilities
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    alpha = [1] * iterations
    beta = [1] * iterations

    averageReward = 0

    if graph:
        tArray = np.arange(0,iterations)
        nMatrix = np.matrix([[0]*iterations]*len(N))
    
    for t in range(iterations):
        if drift:
            probabilities = hf.get_probabilities(-0.001,t)
            if t == 3000 and reset==True:
                alpha = [1] * iterations
                beta = [1] * iterations
        else:
            probabilities = hf.get_probabilities()

        averageReward += thompsonSampleStep(Q, N, probabilities, alpha, beta, calculateReward=True)
        
        if graph:
            for i in range(len(N)):
                nMatrix.itemset((i,t),N[i])

    averageReward /= iterations
    print(f"Thompson Sampling: \n Average Reward: {averageReward}")
    if graph:
        hf.plotMatrix(nMatrix,tArray,f"Convergence Rate for Thompson Sampling")
    else:
        return np.argmax(N)
    

def thompsonSampleStep(Q,N,probabilities,alpha,beta, calculateReward=False):
    index = np.argmax(Q)
    result = probabilities[index]
    
    if result > 0:
        alpha[index] += 1
    else:
        beta[index] += 1
    Q[index] = np.random.beta(alpha[index],beta[index])
    N[index]+= 1
    if calculateReward:
        return result
