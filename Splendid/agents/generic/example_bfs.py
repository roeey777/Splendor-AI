# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Partially implements a greedy breadth-first search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time
from copy import deepcopy
from collections import deque

THINKTIME = 0.95


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Generates actions from this state.
def GetActions(state):
    actions = []
    return actions


# Carry out a given action on this state and return any resultant reward score.
def DoAction(state, action):
    reward = 0
    return reward


# Defines this agent.
class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to reward, if any was found.
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        queue      = deque([ (deepcopy(rootstate), []) ])  # Initialise queue. First node = root state and an empty path.

        # Conduct BFS starting from rootstate.
        while len(queue) and time.time()-start_time < THINKTIME:
            state, path = queue.popleft() # Pop the next node (state, path) in the queue.
            new_actions = GetActions(state) # Obtain new actions available to the agent in this state.
            
            for a in new_actions: # Then, for each of these actions...
                next_state = deepcopy(state)         # Copy the state.
                next_path  = path + [a]              # Add this action to the path.
                reward     = DoAction(next_state, a) # Carry out this action on the state, and note any reward.
                
                if reward:
                    return next_path[0] # If this action resulted in a reward, return the initial action that led there.
                else:
                    queue.append((next_state, next_path)) # If no reward, simply add this state and its path to the queue.
        
        return actions[0] # If no reward was found in the time limit, return the first available action.
        
    
# END FILE -----------------------------------------------------------------------------------------------------------#