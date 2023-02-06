import epsilonGreedy as ep
import thompsonSampling as thom
import sys

def main():
    if sys.argv[1] == "-e":
        handleEpsilonArguments(False)
    elif sys.argv[1] == "-e2":
        handleEpsilonArguments(True)
    elif sys.argv[1] == "-t":
        handleThompsonArguments(drift=False)
    elif sys.argv[1] == "-t2":
        reset = input("Do you want the Thompson Sampling to reset at t = 3000? (y/n): ")
        if reset == "y" or reset == "Y":
            handleThompsonArguments(drift=True,reset=True)
        else:
            handleThompsonArguments(drift=True)
    elif sys.argv[1] == "-c":
        ep.epsilonGreedy(0.11023, graph=False)
        thom.thompsonSample(graph=False)
    elif sys.argv[1] == "-c2":
        reset = input("Do you want the Thompson Sampling to reset at t = 3000? (y/n): ")
        if reset == "y" or reset == "Y":
            ep.epsilonGreedy(0.11023, graph=False, drift=True)
            thom.thompsonSample(graph=False, drift=True, reset=True)
        else:
            ep.epsilonGreedy(0.11023, graph=False, drift=True)
            thom.thompsonSample(graph=False, drift=True)
    elif sys.argv[1] == "-o":
        ep.averageOptimalEpsilon(100,100)
    else:
        print("Argument Not Recognized -- Please Try Again")

def handleEpsilonArguments(drift):
    graph = input("Do you want graphs of the convergences? (y/n): ")
    if graph == "y" or graph == "Y": 
        ep.epsilonGreedy(0.01,drift=drift)
        ep.epsilonGreedy(0.05,drift=drift)
        ep.epsilonGreedy(0.1,drift=drift)
        ep.epsilonGreedy(0.4,drift=drift)
    else:
        ep.epsilonGreedy(0.01, graph=False,drift=drift)
        ep.epsilonGreedy(0.05, graph=False,drift=drift)
        ep.epsilonGreedy(0.1, graph=False,drift=drift)
        ep.epsilonGreedy(0.4, graph=False,drift=drift)

def handleThompsonArguments(drift,reset=False):
    graph = input("Do you want graphs of the convergences? (y/n): ")
    if graph == "Y" or graph == "y":
        thom.thompsonSample(drift=drift,reset=reset)
    else:
        thom.thompsonSample(graph=False,drift=drift,reset=reset)

def testThompsonVailidity():
    count = 0
    badArray = []
    for i in range(100):
        result = thom.thompsonSample(1000,False)
        if result != 2 and result != 17:
            count += 1
            badArray.append(result)

    print(count)
    print(badArray)
    


    




 
if __name__=="__main__":
    main()
