# Running the Program

To run the program, type `python HW2_Watson_Andrew.py`, followed by one of the following command line arguments:

- "-e": This argument will run the epsilon greedy algorithm for 10,000 steps with epsilons of 0.01, 0.05, 0.1, and 0.4. 
- "-e2": This argument will do the same as "-e", but for the Part 2 Moving Bandits.
- "-t": This argument will run the Thompson Sampling for 10,000 iterations. 
- "-t2": This argument will do the same as "-t", but for the Part 2 Moving Bandits
- "-c": This argument will run the epsilon greedy algorithm with the optimal epsilon of 0.11023 (See "Finding Optimal Epsilon" below) and then will run the Thompson Sampling algorithm. The average rewards of each will be printed to the command line. 
- "-c2": This argument will do the same as "-c", but for the Part 2 Moving Bandits. 
- "-o": This argument will find the average optimal epsilon from 0 to 1 in steps of 0.01 with 100 iterations. 

# Part 1

## Epsilon-Greedy

- Instantiate Q: an array of expected results -- each index represents an arm
- Instantiate N: an array of how many times each result has been chosen -- each index represents an arm
- Pick a random number x
    - If x > epsilon, take the action with the best Q
    - Else, pick a random action
- Update N and Q for the selected index
    - `N[index] += 1`
    - `Q = Q + 1/N * (r-Q)`

## Finding Optimal Epsilon

- The algorithm is said to have "converged" at time step t if 80% of the total choices the algorithm has made up until t have been the optimal arm. 
- The "convergence speed" of the algorithm is the number of iterations (timesteps) t that it takes for the algorithm to converge. A minimum t will be a maximum convergence speed. 
- Start at epsilon = 0, and compute the convergence speed for a sample of epsilons (my algorithm takes the number of steps as an argument. For efficiency, I ran the algorithm with 100 steps for epsilon, i.e. 0.00, 0.01, 0.02, etc).
    - Take the best convergence speed
    - The epsilon which maximizes convergence speed is different every time. It depends on what the random distributions give.
    - Therefore, we can compute the average optimal epsilon. 
        - Compute optimal epsilon for some number of iterations, then divide by the total number of iterations
        - This takes a large amount of time
- The average optimal epsilon with 100 steps and 1000 iterations was found to be epsilon = 0.11023
    - If you want to run the average optimal epsilon algorithm on your computer, please don't use 1000 iterations. It takes a LONG time. 

## Thompson Sampling 
- Q for each action follows a Beta distribution
- alpha is the amount of times we succeeded in getting a reward
- beta is the number of times we failed in getting a reward
- Take the action which maximizes Q
- Update alpha and beta
    - If the actual result is positive, we succeeded in getting a reward, so alpha += result
    - If the actual result is negative, we failed to get a reward, so beta -= result

## Comparison: Optimal Epsilon vs Thompson Sampling
- Thompson Sampling does not always result in the Arm with the highest mean being picked the most


# Part 2

## Changes

- The adjustment to the algorithm is just adding optional "drift" and "t" parameters to the get_probabilities() function. The probability changes will be handled within that method. In the Thompson Sampling and Epsilon Greedy functions, simply pass in drift = -0.001 and t. 
- To restart the Thompson Sampling at 3000, add a piece of code that resets alpha and beta when t=3000.

## Results

- The Epsilon Greedy algorithm has a very hard time converging to the new value in general
- The Thompson Sampling converges faster if you restart the algorithm when a large change (i.e. t=3000) occurs. 
