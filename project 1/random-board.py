# Naomi Igbinovia 
# CSCI 4350 -- OLA1 
# Joshua Phillips 
# October 2, 2023

import sys
import numpy.random as random

# the seed number and number random moves are checked for in the command-line 
if len(sys.argv) != 3:
    print("Your input was incorrect. Please enter it as the following: ")
    print("python3 random-board.py [seed number] [number of random moves]")
   # print("Usage: %s [seed] [number of random moves]" % sys.argv[0])
    sys.exit(1)

# this function randomizes the base board using the user given seed number and 
# number of random moves 
def randomize_board(seed, num_moves):

# a random number generator with the user seed is initialized 
    rng = random.default_rng(seed)

# the initial board is created 
    board = list(range(9))

# the user number of random moves are performed 
    for _ in range(num_moves):

# random moves are generated: 0 for up, 1 for down, 2 for left, and 3 for right 
        move = rng.integers(4)

# the blank tile's index is found 
        blank_index = board.index(0)

# the blank tile is moved up if possible 
        if move == 0:  
            if blank_index > 2:
                board[blank_index], board[blank_index - 3] = board[blank_index - 3], board[blank_index]

# the blank tile is moved down if possible 
        elif move == 1: 
            if blank_index < 6:
                board[blank_index], board[blank_index + 3] = board[blank_index + 3], board[blank_index]
        
# the blank tile is moved left if possible 
        elif move == 2:  
            if blank_index % 3 > 0:
                board[blank_index], board[blank_index - 1] = board[blank_index - 1], board[blank_index]
        
# the blank tile is moved right if possible    
        elif move == 3:  
            if blank_index % 3 < 2:
                board[blank_index], board[blank_index + 1] = board[blank_index + 1], board[blank_index]

    return board

# this function prints the board configuration in a readable format 
def print_board(board):
    for i in range(3):
        print(board[i * 3], board[i * 3 + 1], board[i * 3 + 2])

def main():

# the command-line arguments are parsed 
    seed = int(sys.argv[1])
    num_moves = int(sys.argv[2])

# a random board configuration is generated and printed 
    randomized_board = randomize_board(seed, num_moves)
#    print("Randomized Board after", num_moves, "moves:")
    print_board(randomized_board)

main()