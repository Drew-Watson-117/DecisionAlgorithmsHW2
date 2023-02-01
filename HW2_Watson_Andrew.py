import helperFunctions as hf
import numpy as np
import matplotlib.pyplot as plt

def main():

    probabilities = hf.get_probabilities()
    Q = [0] * len(probabilities) # List of EXPECTED probabilities
    N = [0] * len(probabilities) # Number of times each arm was selected BEFORE timestep t
    iterations = 10000
    epsilon = 0.6

    # tArray = [0]*iterations
    # nMatrix = [[0]*iterations]*len(N)
    for t in range(iterations):
        probabilities = hf.get_probabilities()
        epsilonGreedy(epsilon,probabilities,Q,N)
    print(N)
    print(Q)
    # plt.title(f"Convergence Rate for epsilon={epsilon}")
    # plt.xlabel("t (iterations)")
    # plt.ylabel("Number of times it was selected")
    # for i in range(len(N)):
    #     plt.plot(tArray,nMatrix[i], label=f"Arm {i}")
    # plt.legend()
    # plt.show()

def epsilonGreedy(epsilon,probabilities,Q,N):
    if np.random.random() < epsilon:
        # Explore
        index = np.random.randint(0,len(probabilities))
    else:
        # Exploit
        index = np.argmax(Q)

    action = probabilities[index]
    # Update chosen action
    N[index] += 1 # Might need to be after updateExpected -- Then get a /0 error
    Q[index] = hf.updateExpected(action, index, N[index])


 
if __name__=="__main__":
    main()
