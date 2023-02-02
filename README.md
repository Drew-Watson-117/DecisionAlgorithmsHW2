# epsilon-greedy

- Instantiate Q: an array of expected results -- each index represents an arm
- Instantiate N: an array of how many times each result has been chosen -- each index represents an arm
- Pick a random number x
    - If x > epsilon, take the action with the best Q
    - Else, pick a random action
- Update N and Q for the selected index
    - `N[index] += 1`
    - `Q = Q + 1/N * (r-Q)`

# Thompson Sampling 
- Q for each action follows a Beta distribution
- alpha is the amount of times we succeeded in getting a reward
- beta is the number of times we failed in getting a reward
- Take the action which maximizes Q
- Update alpha and beta
    - If the actual result is positive, we succeeded in getting a reward, so alpha += result
    - If the actual result is negative, we failed to get a reward, so beta -= result

