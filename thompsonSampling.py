import helperFunctions as hf
import numpy as np


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
    print(f"Thompson Sampling: \n Expected Values for Arms:\n {Q}\n Number of times chosen: \n {N}")
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