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

def get_probabilities(drift=0, t=0):
    
    probs = [
        np.random.normal(0+drift*t, 5),     #0
        np.random.normal(-0.5+drift*t,12),  #1
        np.random.normal(2+drift*t,3.9),    #2
        np.random.normal(-0.5+drift*t,7),   #3
        np.random.normal(-1.2+drift*t,8),   #4
        np.random.normal(-3+drift*t,7),     #5
        np.random.normal(-10+drift*t,20),   #6
        np.random.normal(-0.5+drift*t,1),   #7
        np.random.normal(-1+drift*t,2),     #8
        np.random.normal(1+drift*t,6),      #9
        np.random.normal(0.7+drift*t,4),    #10
        np.random.normal(-6+drift*t,11),    #11
        np.random.normal(-7+drift*t,1),     #12
        np.random.normal(-0.5+drift*t,2),   #13
        np.random.normal(-6.5+drift*t,1),   #14
        np.random.normal(-3+drift*t,6),     #15
        np.random.normal(0+drift*t,8),      #16
        np.random.normal(2+drift*t,3.9),    #17
        np.random.normal(-9+drift*t,12),    #18
        np.random.normal(-1+drift*t,6),     #19
        np.random.normal(-4.5+drift*t,8)    #20        
    ]
    if t >= 3000:
        probs[0] = np.random.normal(5+drift*t, 5)
        probs[2] = np.random.normal(2-0.5+drift*t,3.9)
        probs[7] = np.random.normal(3-0.5+drift*t,1)
        np.random.normal(3-9+drift*t,12)
    
    
    return probs
