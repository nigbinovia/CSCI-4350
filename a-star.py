# Naomi Igbinovia 
# CSCI 4350 -- OLA1 
# Joshua Phillips 
# October 2, 2023

import sys

# this Node class represents a node in a search tree that will be used one of the 
# heuristic problems 
class Node:
    # a counter is given to each node created in this class to pair with its unquie ID
    node_id_counter = 0

    # this constructor initializes a node in the search tree, passing in a list of nodes 
    # representing the problem state, a dictionary holding the nodes' positions, and the 
    # node's parent in the search tree
    def __init__(self, nodes, node_positions = None, parent = None):

        # the current count for the counter is assigned as the node's id and the counter 
        # is incremented up by one for the next potential node 
        self.node_id = Node.node_id_counter
        Node.node_id_counter += 1
        
        # the problem's state presented by the current list of nodes is stored here 
        self.nodes = nodes

        # the current number of nodes is calculated and stored here 
        N = len(self.nodes)

        # if the current node doesn't have a set of positions stored, the find_node_positions 
        # method is called to find and assign the postitions to the current node. otherwise, 
        # the given node positions are used 
        if node_positions is None:
            self.node_positions = self.find_node_positions(self.nodes)
        else:
            self.node_positions = node_positions

        # the cost of reaching the current node is initalized 
        self.g = 0
        # the parent node is stored 
        self.parent = parent

        # if the current node has a parent node already, the cost is calculated from the starting 
        # node and the cost is incremented from the parent up 1 
        if self.parent:
            self.g = self.parent.g + 1

    # this method finds the nodes in the problem state, passing in a list of nodes representing 
    # the problem state
    def find_node_positions(self, nodes):

        # a dictionary is initalized to store the current node's positions
        node_positions = {}  

        # for each row (y-axis) of the nodes,
        for y in range(len(nodes)):

            # for each column (x-axis) of the nodes,
            for x in range(len(nodes[y])):

                # the position of the node is stored in a (y,x) format with its value as the key 
                node_positions[nodes[y][x]] = (y, x)
        return node_positions

    # this method checks if a move in a given direction is valid, passing in a the direction 
    # of the move (either "up", "down", "left", or "right")
    def is_valid_move(self, direction):

        # the blank tile's current position is obtained
        blank_tile_ypos, blank_tile_xpos = self.node_positions[0]

        # the move is checked if it's valid and blank tile can move up without being outside of 
        # bounds
        if direction == "up":
            return blank_tile_ypos > 0
            
        # the move is checked if it's valid and blank tile can move down without being outside 
        # of bounds
        elif direction == "down":
            return blank_tile_ypos < len(self.nodes) - 1

        # the move is checked if it's valid blank tile can move left without being outside of bounds
        elif direction == "left":
            return blank_tile_xpos > 0

        # the move is checked if it's valid blank tile can move right without being outside of bounds
        elif direction == "right":
            return blank_tile_xpos < len(self.nodes[0]) - 1

    # this method moves the blank tile in a certain direction if the move is valid, passing 
    # in a the direction of the move
    def move_blank(self, direction):

        # the move is checked if it's valid 
        if not self.is_valid_move(direction):
            return None

        # the blank tile's current position is obtained
        blank_tile_ypos, blank_tile_xpos = self.node_positions[0]

        # the blank tile's new position is calculated based on the given direction 
        new_blank_tile_ypos, new_blank_tile_xpos = blank_tile_ypos, blank_tile_xpos

        if direction == "up":
            new_blank_tile_ypos -= 1
        elif direction == "down":
            new_blank_tile_ypos += 1
        elif direction == "left":
            new_blank_tile_xpos -= 1
        elif direction == "right":
            new_blank_tile_xpos += 1

        # a copy of the current nodes is created to modify 
        new_nodes = [row[:] for row in self.nodes]

        # the positions of blank tile and tile it's moving to are swapped with each other 
        new_nodes[blank_tile_ypos][blank_tile_xpos], new_nodes[new_blank_tile_ypos][new_blank_tile_xpos] = (
            new_nodes[new_blank_tile_ypos][new_blank_tile_xpos],
            new_nodes[blank_tile_ypos][blank_tile_xpos],
        )

        # a new node is created to represent the updated state 
        return Node(
            new_nodes,
            node_positions = self.find_node_positions(new_nodes),
            parent = self,
        )

    # this method creates a deep copy of the current node 
    def copy(self):

        # a new node with copied nodes, positons, and parent is created 
        return Node(

            # a deep copy of the nodes is created 
            [row[:] for row in self.nodes],

            # a copy of the node positions is created 
            node_positions = self.node_positions.copy(),

            # the parent node is referenced 
            parent = self.parent,
        )

    # this method generates the child nodes of the current by moving the blank tile in all 
    # possible directions 
    def generate_child_nodes(self):

        # a list to store the child nodes is initalized 
        child_nodes = []

        # for all the possible directions, 
        for direction in ["up", "down", "left", "right"]:

            # if the move in the current direction is valid, the blank tile is moved in the current 
            # direction to generate a new node and the new node is added to the list 
            if self.is_valid_move(direction):
                new_node = self.move_blank(direction)
                child_nodes.append(new_node)
        return child_nodes

    # this method compares nodes for less than based on their f values (g + h), and if their 
    # f values are equal, they're compared by their node IDs
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h) or ((self.g + self.h) == (other.g + other.h) and self.node_id < other.node_id)

    # this method checks if nodes are equal to each other based on their states
    def __eq__(self, other):
        return self.nodes == other.nodes

    # this method computes the hash value of the current node based on its current state
    def __hash__(self):
        return hash(str(self.nodes))

    # this method gets the string representtation of the current node's state
    def __str__(self):
        return "\n".join(
            [" ".join(str(tile) if tile != 0 else "0" for tile in row) for row in self.nodes]
        )


# this Explore class explores a given heuristic puzzle by findung a sequence of 
# moves from a given start state to the goal state 
class Explore:

    # this constructor initializes the explore puzzle solver with a start and goal 
    # state with the start and goal states that are passed in 
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state


    # this method solves the puzzle problem using a basic search algorithm 
    def solve_base_case(self):

        # the open list (storing potential expanded nodes), closed list (storing 
        # the visited states), visited (storing the number of nodes expanded), 
        # and depth (storing the solution's depth) are declared and defined
        open_list = []
        closed_list = set()
        visited = 0
        depth = 0

        # the start state is added to the open list with its cost 
        open_list.append((self.start_state.g, self.start_state))

        # a flag called found is initialized to false until the goal state is found 
        found = False

        # while the search is going, 
        while open_list:
    
            # the open list is sorted by cost, and the next node is obtained from the 
            # open list
            open_list.sort(key=lambda x: x[0])  
            current_cost, current = open_list.pop(0)  
        
        
            # if the current node's state matches the goal state, 
            if current == self.goal_state:

                # the depth, closed list, and found flag are updated, and the loop is 
                # ended to print the data
                depth = current.g
                closed_list.add(current)
                found = current
                break

            # if the current node is in the closed list, skip it 
            if current in closed_list:
                continue

            # the current node otherwise is added to the closed list and the visited 
            # count is incremented by 1
            closed_list.add(current)
            visited += 1
        
            # for every potential child of the current node 
            for child in current.generate_child_nodes():
        
                # the child's cost is its own cost (since we're not using any heuristic)
                child_cost = child.g
            
                # the child node is added to the open list with its cost
                open_list.append((child_cost, child))
            

        # the statistics (the total nodes visited [V], the maximum nodes in memory [N], 
        # the depth of the solution [d], and the branching factor [b]) are printed to 
        # the user 
        print("V =", visited)
        print("N =", Node.node_id_counter)
        print("d =", depth)
        print("b =", round((Node.node_id_counter) ** (1 / depth), 5))
        print()

        # if the solution is found, the path is printed 
        if found:
            path = []
            while found:
                path.append(found)
                found = found.parent
            path.reverse()

            for node in path:
                print(node)
                print()
        else:
            print("No solution found")


    # this method solves the puzzle problem using the misplaced tiles heuristic 
    def solve_misplaced_tiles(self):

        # the open list (storing potential expanded nodes), closed list (storing 
        # the visited states), visited (storing the number of nodes expanded), 
        # and depth (storing the solution's depth) are declared and defined
        open_list = []
        closed_list = set()
        visited = 0
        depth = 0

        # the misplaced tiles heuritistic is calculated for the start state
        f = 0
        for y in range(len(self.start_state.nodes)):
            for x in range(len(self.start_state.nodes[y])):
                if self.start_state.nodes[y][x] == 0:
                    continue
                if self.start_state.nodes[y][x] != self.goal_state.nodes[y][x]:
                    f += 1

        # the start state is added to the opne list with its f value 
        open_list.append((f + self.start_state.g, self.start_state))

        # a flag called found is initialized to false until the goal state is found 
        found = False

        # while the search is going, 
        while open_list:
        
            # the open list is sorted by f values, and the next node is obtained from the 
            # open list
            open_list.sort(key=lambda x: x[0])  
            current_f, current = open_list.pop(0)  
            
            
            # if the current node's state matches the goal state, 
            if current == self.goal_state:

                # the depth, closed list, and found flag are updated, and the loop is 
                # ended to print the data
                depth = current.g
                closed_list.add(current)
                found = current
                break

            # if the current node is the closed list, skip it 
            if current in closed_list:
                continue

            # the current node otherwise is added to the closed list and the visited 
            # count is incremented by 1
            closed_list.add(current)
            visited += 1
            
            # for every potential child of the current node 
            for child in current.generate_child_nodes():
            
            # the misplaced tile heurisitic is recalculated for the 
            # potential child node and is added to the open list with its f
            # value 
                f = 0
                for y in range(len(child.nodes)):
                    for x in range(len(child.nodes[y])):
                        if child.nodes[y][x] == 0:
                            continue
                        if child.nodes[y][x] != self.goal_state.nodes[y][x]:
                            f += 1
                open_list.append((f + child.g, child))
                

        # the statistics (the total nodes visited [V], the maximum nodes in memory [N], 
        # the depth of the solution [d], and the branching factor [b]) are printed to 
        # the user 
        print("V =", visited)
        print("N =", Node.node_id_counter)
        print("d =", depth)
        print("b =", round((Node.node_id_counter) ** (1 / depth), 5))
        print()

        # if the solution is found, the path is printed 
        if found:
            path = []
            while found:
                path.append(found)
                found = found.parent
            path.reverse()

            for node in path:
                print(node)
                print()
        else:
            print("No solution found")

    # this method solves the puzzle problem using the manhattan distance heuristic 
    def solve_manhattan_distance(self):

        # the open list (storing potential expanded nodes), closed list (storing 
        # the visited states), visited (storing the number of nodes expanded), 
        # and depth (storing the solution's depth) are declared and defined
        open_list = []
        closed_list = set()
        visited = 0
        depth = 0

        # the manhattan distance heuritistic is calculated for the start state
        f = 0  
        for y in range(len(self.start_state.nodes)):
            for x in range(len(self.start_state.nodes[y])):
                if self.start_state.nodes[y][x] != 0:
                    goal_state_tile = self.goal_state.node_positions[self.start_state.nodes[y][x]]
                    f += abs(y - goal_state_tile[0]) + abs(x - goal_state_tile[1])
        f += self.start_state.g  

        # the start state is added to the opne list with its f value 
        open_list.append((f, self.start_state))
        
        # a flag called found is initialized to false until the goal state is found 
        found = False

        # while the search is going, 
        while open_list:

            # the open list is sorted by f values, and the next node is obtained from the 
            # open list 
            open_list.sort(key=lambda x: x[0])  
            current_f, current = open_list.pop(0)  
            
            # if the current node's state matches the goal state, 
            if current == self.goal_state:

                # the depth, closed list, and found flag are updated, and the loop is 
                # ended to print the data
                depth = current.g
                closed_list.add(current)
                found = current
                break

            # if the current node is the closed list, skip it 
            if current in closed_list:
                continue

            # the current node otherwise is added to the closed list and the visited count
            # is incremented by 1 
            closed_list.add(current)
            visited += 1
            
            # for every potential child of the current node 
            for child in current.generate_child_nodes():

                # the manhattan distance heurisitic is recalculated for the 
                # potential child node and is added to the open list with its f
                # value 
                f = 0  
                for y in range(len(child.nodes)):
                    for x in range(len(child.nodes[y])):
                        if child.nodes[y][x] != 0:
                            goal_state_tile = self.goal_state.node_positions[child.nodes[y][x]]
                            f += abs(y - goal_state_tile[0]) + abs(x - goal_state_tile[1])

                open_list.append((f + child.g, child))
               

        # the statistics are printed to the user 
        print("V =", visited)
        print("N =", Node.node_id_counter)
        print("d =", depth)
        print("b =", round((Node.node_id_counter) ** (1 / depth), 5))
        print()

        # if the solution is found, the path is printed 
        if found:
            path = []
            while found:
                path.append(found)
                found = found.parent
            path.reverse()

            for node in path:
                print(node)
                print()
        else:
            print("No solution found")

    # this method solves the puzzle problem using the manhattan distance heuristic 
    # combined with the linear conflicts heurisitic
    def solve_manhattan_linear_conflict(self):

        # the open list (storing potential expanded nodes), closed list (storing 
        # the visited states), visited (storing the number of nodes expanded), 
        # and depth (storing the solution's depth) are declared and defined
        open_list = []
        closed_list = set()
        visited = 0
        depth = 0

        # the manhattan distance + lienar conflicts heuritistic is calculated for the 
        # start state
        f = 0  
        goal_positions = self.goal_state.find_node_positions(self.goal_state.nodes)

        for y in range(len(self.start_state.nodes)):
            for x in range(len(self.start_state.nodes[y])):
                tile = self.start_state.nodes[y][x]
                if tile == 0:
                    continue
                goal_y, goal_x = goal_positions[tile]
                if (y == goal_y) and (x == goal_x):
                    continue  
                f += abs(y - goal_y) + abs(x - goal_x)
                if (y == goal_y) or (x == goal_x):
                    for i in range(len(self.start_state.nodes)):
                        for j in range(len(self.start_state.nodes[i])):
                            if (i != y or j != x) and self.start_state.nodes[i][j] != 0 and goal_positions[self.start_state.nodes[i][j]][1] == goal_x and (i == goal_y or (y < goal_y < i) or (y > goal_y > i)):
                                f += 2
        f += self.start_state.g  

        # the start state is added to the opne list with its f value 
        open_list.append((f, self.start_state))
        
        # a flag called found is initialized to false until the goal state is found 
        found = False

        # while the search is going,
        while open_list:

            # the open list is sorted by f values, and the next node is obtained from the 
            # open list
            open_list.sort(key=lambda x: x[0])  
            current_f, current = open_list.pop(0)  
            

            # if the current node's state matches the goal state, 
            if current == self.goal_state:

                # the depth, closed list, and found flag are updated, and the loop is 
                # ended to print the data
                depth = current.g
                closed_list.add(current)
                found = current
                break

            # if the current node is the closed list, skip it 
            if current in closed_list:
                continue

            # the current node otherwise is added to the closed list and the visited count 
            # is incremented by 1 
            closed_list.add(current)
            visited += 1

            # for every potential child of the current node 
            for child in current.generate_child_nodes():

                # the manhattan distance + linear conflicts heurisitic is 
                # recalculated for the potential child node and is added 
                # to the open list with its f value 
                f = 0  
                goal_positions = self.goal_state.find_node_positions(self.goal_state.nodes)

                for y in range(len(child.nodes)):
                    for x in range(len(child.nodes[y])):
                        tile = child.nodes[y][x]
                        if tile == 0:
                            continue
                        goal_y, goal_x = goal_positions[tile]
                        if (y == goal_y) and (x == goal_x):
                            continue  
                        f += abs(y - goal_y) + abs(x - goal_x)
                        if (y == goal_y) or (x == goal_x):
                            for i in range(len(child.nodes)):
                                for j in range(len(child.nodes[i])):
                                    if (i != y or j != x) and child.nodes[i][j] != 0 and goal_positions[child.nodes[i][j]][1] == goal_x and (i == goal_y or (y < goal_y < i) or (y > goal_y > i)):
                                        f += 2

                open_list.append((f + child.g, child))

        # the statistics are printed to the user 
        print("V =", visited)
        print("N =", Node.node_id_counter)
        print("d =", depth)
        print("b =", round((Node.node_id_counter) ** (1 / depth), 5))
        print()

        # if the solution is found, the path is printed 
        if found:
            path = []
            while found:
                path.append(found)
                found = found.parent
            path.reverse()

            for node in path:
                print(node)
                print()
        else:
            print("No solution found")


# this method parses the start state given as a string and creates a Node object 
# representing the start state 
def parse_start_state(start_state_str):

    # the start state string is split and converted into a list of integers
    start_state_input = list(map(int, start_state_str.split()))

    # a Node object representing the start state is created 
    start_state = Node([start_state_input[:3], start_state_input[3:6], start_state_input[6:]])
    return start_state

# the main function parses command-line arguments and solves the puzzle problem 
def main():

    # if an incorrect number of command-line arguments is provided, an error statement 
    # is displayed and a correct input example is prompted to the user
    if len(sys.argv) != 3:
        print("Your input was incorrect. Please enter it as the following: ")
        print("python3 astar.py '<start_state>' '<heuristic>'")
        sys.exit(1)

    # the goal state of for all the puzzle problems is defined 
    goal_state = Node([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # the state state and the selected heurisitic choice are stored from the 
    # command-line argument 
    start_state_str = sys.argv[1]
    heuristic_choice = sys.argv[2]

    # if the heuristic choice given is invalid, an error statement is displayed and the 
    # the correct options are prompted to the user 
    if heuristic_choice not in ["0", "1", "2", "3"]:
        print("Your heurisitic choice is invalid. Please enter one of the following: ")
        print ("0 -- A* Star No Heuristic")
        print ("1 -- Misplaced Tiles Heuristic")
        print ("2 -- Manhattan Distance Heuristic")
        print ("3 -- Manhattan Distance + Linear Conflicts Heuristic")
        sys.exit(1)

    # the start state string is parsed to create a Node object representing the start state 
    start_state = parse_start_state(start_state_str)

    # a Explore object is created with the start and goal states
    solver = Explore(start_state, goal_state)

    # the puzzle is solved based on the chosen heurisitic
    # if 0 is chosen, the A* Star No Heuristic is called 
    if heuristic_choice == "0":
        solver.solve_base_case()

    # if 1 is chosen, the Misplaced Tiles Heuristic is called 
    elif heuristic_choice == "1":
        solver.solve_misplaced_tiles()
    
    # if 2 is chosen, the Manhattan Distance Heuristic is called 
    elif heuristic_choice == "2":
        solver.solve_manhattan_distance()
    
    # if 3 is chosen, the Manhattan Distance + Linear Conflicts Heuristic is called 
    elif heuristic_choice == "3":
        solver.solve_manhattan_linear_conflict()

if __name__ == "__main__":
    main()