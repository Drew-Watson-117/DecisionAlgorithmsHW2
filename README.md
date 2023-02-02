# Epsilon-Greedy

- Instantiate Q: an array of expected results -- each index represents an arm
- Instantiate N: an array of how many times each result has been chosen -- each index represents an arm
- Pick a random number x
    - If x > epsilon, take the action with the best Q
    - Else, pick a random action
- Update N and Q for the selected index
    - `N[index] += 1`
    - `Q = Q + 1/N * (r-Q)`

# Finding Optimal Epsilon

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

# Thompson Sampling 
- Q for each action follows a Beta distribution
- alpha is the amount of times we succeeded in getting a reward
- beta is the number of times we failed in getting a reward
- Take the action which maximizes Q
- Update alpha and beta
    - If the actual result is positive, we succeeded in getting a reward, so alpha += result
    - If the actual result is negative, we failed to get a reward, so beta -= result

