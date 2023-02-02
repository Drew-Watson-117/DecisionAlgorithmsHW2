import numpy as np
import matplotlib.pyplot as plt


def updateExpected(realValue, expectedValue, N):
    return expectedValue + (1 / (N+1)) * (realValue - expectedValue)

def plotMatrix(matrix, array,title):
    matrix = matrix.tolist()
    array = array.tolist()

    plt.title(title)
    plt.xlabel("t (iterations)")
    plt.ylabel("Number of times it was selected")
    for i in range(len(matrix)):
        plt.plot(array,matrix[i], label=f"Arm {i}")
    plt.legend()
    plt.show()

def get_probabilities(drift=0):
    
    probs = [
        np.random.normal(0, 5),
        np.random.normal(-0.5,12),
        np.random.normal(2,3.9),
        np.random.normal(-0.5,7),
        np.random.normal(-1.2,8),
        np.random.normal(-3,7),
        np.random.normal(-10,20),
        np.random.normal(-0.5,1),
        np.random.normal(-1,2),
        np.random.normal(1,6),
        np.random.normal(0.7,4),
        np.random.normal(-6,11),
        np.random.normal(-7,1),
        np.random.normal(-0.5,2),
        np.random.normal(-6.5,1),
        np.random.normal(-3,6),
        np.random.normal(0,8),
        np.random.normal(2,3.9),
        np.random.normal(-9,12),
        np.random.normal(-1,6),
        np.random.normal(-4.5,8)              
    ]
    
    return probs