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
        ep.epsilonGreedy(0.01, graph=False)
        thom.thompsonSample(graph=False)
    elif sys.argv[1] == "-c2":
        reset = input("Do you want the Thompson Sampling to reset at t = 3000? (y/n): ")
        if reset == "y" or reset == "Y":
            ep.epsilonGreedy(0.01, graph=False, drift=True)
            thom.thompsonSample(graph=False, drift=True, reset=True)
        else:
            ep.epsilonGreedy(0.01, graph=False, drift=True)
            thom.thompsonSample(graph=False, drift=True)
    elif sys.argv[1] == "-o":
        ep.mostOptimalEpsilon(100)
        ep.mostOptimalEpsilon(100, fastest=False) #Optimize reward

    elif sys.argv[1] == "-o2":
        ep.mostOptimalEpsilon(100,drift=True)
        ep.mostOptimalEpsilon(100, drift=True, fastest=False) #Optimize reward

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
    


 
if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Error: Must supply exactly 1 command line argument to the program")
    else:
        main()