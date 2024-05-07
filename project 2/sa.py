# Naomi Igbinovia 
# CSCI 4350 -- OLA2
# Joshua Phillips 
# October 18, 2023

import SumofGaussians as SG
import numpy as np
import sys

# three command-line arguments are taken in: the random seed number (seed), 
# the number of dimensions (dims) for the soG function, and the number of 
# Gaussians (ncenters) for the soG function 
seed = int(sys.argv[1])
dims = int(sys.argv[2])
ncenters = int(sys.argv[3])

# a random number is created using the seed variable
rng = np.random.default_rng(seed)

# the Sum of Gaussians object is initialized with the given parameters 
sog = SG.SumofGaussians(dimensions = dims, number_of_centers = ncenters, rng = rng)

# the tolerance is declared 
epsilon = 1e-8

# the initial point is created using a random location in the range of
# 0 to 10
x = rng.uniform(size = dims) * 10.0

# the max iterations and temperature are declared 
max_iterations = 100000
temperature = 1.0
 
def acceptance(probability, temperature):
    if probability > 0:
        return np.exp(-probability / temperature)
    else:
        return 1.0
        
# the iteration count is started 
iteration = 0

# while the max amount of iterations haven't been reached,
while iteration < max_iterations:
    
# a new location is generated
    y = x + rng.uniform(low =- 0.05, high = 0.05, size = dims)
    
# the function values are evaluted 
    sog_value_x = sog.Evaluate(x)
    sog_value_y = sog.Evaluate(y)
    
# the energy difference is calculated 
    probability = sog_value_y - sog_value_x
    
# the metropolis criterion is performed
    if probability > 0 or acceptance(probability, temperature) > rng.random():
        x = y
    
# the tempeature is reduced according the annealing schedule 
    temperature *= 0.999  
    
# the location and the SoG function value of the given step is printed 
    print(f"{str(x).lstrip('[').rstrip(']')} {sog_value_x:.8f}")
    
# if convergence has occurred, then the loop stops
    if temperature < 1e-8:
        break

# otherwise, it moves on to the next iteration 
    iteration += 1

# the final location and the SoG function value of the final step is printed 
sog_value_final = sog.Evaluate(x)
print(f"{str(x).lstrip('[').rstrip(']')} {sog_value_final:.8f}")

