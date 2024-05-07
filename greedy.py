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

# the max iterations and step size are declared 
max_iterations = 100000
step = 0.01

# the iteration count is started 
iteration = 0

# while the max amount of iterations haven't been reached, 
while iteration < max_iterations:

# the gradient ascent is performed 
    gradient = sog.Gradient(x)
    new_x = x + step * gradient

# the function values are evaluted
    old_sog_value = sog.Evaluate(x)
    new_sog_value = sog.Evaluate(new_x)

# the location and the SoG function value of the given step is printed 
    print(f"{str(x).lstrip('[').rstrip(']')} {old_sog_value:.8f}")
    
# if convergence has occurred, then the loop stops
    if abs(new_sog_value - old_sog_value) < epsilon:
        break
        
# otherwise, it moves on to the next iteration 
    x = new_x
    iteration += 1
    
# the final location and the SoG function value of the final step is printed 
print(f"{str(x).lstrip('[').rstrip(']')} {new_sog_value:.8f}")
