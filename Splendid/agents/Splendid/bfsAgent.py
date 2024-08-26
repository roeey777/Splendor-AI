import time
# from copy import deepcopy, copy
from collections import deque
import json, traceback
THINKTIME = 0.75

from Splendor.splendor_utils import *
from Splendor.splendor_model import SplendorGameRule, Card, SplendorState

def handleCard(cardDict):
    card = Card(cardDict['colour'],cardDict['code'],cardDict['cost'],cardDict['deck_id'],cardDict['points'])
    return card 

def handleAgent(d):
    s= SplendorState.AgentState(d["id"])
    s.score= d["score"]
    s.gems = d["gems"]
    for colour in d["cards"]:
        cards = d["cards"][colour]
        d["cards"][colour] = list(map(handleCard, cards))
    s.cards = d["cards"]
    s.nobles = d["nobles"]
    s.passed = d["passed"]
    # s.agent_trace = d["agent_trace"]
    s.last_action = d["last_action"]
    return s

def handleBoard(d): 
    b = SplendorState.BoardState(2)
    b.decks = []
    decks = d['decks'] 
    for deck in decks:
        b.decks.append(list(map(handleCard, deck)))
    b.dealt = []
    dealt = d['dealt']
    for row in dealt:
        b.dealt.append(list(map(handleCard, row)))
    b.gems = d['gems']
    b.nobles = d['nobles']
    return b

def handle(d):
    board = handleBoard(d['board'])
    agents = list(map(handleAgent, d["agents"]))
    state = SplendorState(2)
    state.board = board
    state.agents = agents
    state.agent_to_move = d['agent_to_move']
    return state

def copy(state):
    json_string = json.dumps(state, default=lambda x: x.__dict__)
    return handle(json.loads(json_string))

# Carry out a given action on this state and return any resultant reward score.
def DoAction(action):
    reward= 0
    if 'buy' in action['type']:
        card = action['card']
        reward += card.points *2+1

    if action['noble']:
        reward += 3*3
    return reward


# Defines this agent.
class myAgent():

    NUM_PLAYERS= 2
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        self.rules = SplendorGameRule(self.NUM_PLAYERS)

    def getActions(self, state):  # return a list of legal actions
        return self.rules.getLegalActions(state, self.id)

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to reward, if any was found.
    def SelectAction(self, actions, rootstate):
        try:
            start_time = time.time()
            queue = deque([(rootstate, [])])  # Initialise queue. First node = root state and an empty path.

            # Conduct BFS starting from rootstate.
            while len(queue) and time.time() - start_time < THINKTIME:
                state, path = queue.popleft()  # Pop the next node (state, path) in the queue.
                new_actions = self.getActions(state)  # Obtain new actions available to the agent in this state.

                max_reward=0
                path_chosen= path+ [new_actions[0]]

                for a in new_actions:  # Then, for each of these actions...
                    # next_state = copy(state)  # Copy the state.
                    next_path = path + [a]  # Add this action to the path.
                    reward = DoAction(a)  # Carry out this action, and note any reward.

                    if reward> max_reward:
                        max_reward= reward   #update the reward and the path of an action with the highest reward so far
                        path_chosen = next_path
                        
                if max_reward>0:
                    return path_chosen[0]
                else:
                    for a in new_actions:
                        queue.append((self.rules.generateSuccessor(copy(state), a, self.id), path + [a]))  # If no reward, simply add this state and its path to the queue.
            return actions[0]  # If no reward was found in the time limit, return the first available action.
        except Exception as e:
            traceback.print_exc()
            return actions[0]

