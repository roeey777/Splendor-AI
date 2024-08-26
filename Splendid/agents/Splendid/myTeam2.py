from typing import AsyncContextManager
from template import Action, Agent, GameState
from Splendor.splendor_utils import *
import random, itertools
from Splendor.splendor_model import SplendorGameRule, Card
import cProfile, pstats
import json
import numpy as np

STARTUP_TIME = 14 #SECONDS
ROUND_TIME = 1 #SECOND
SEARCH_DEPTH = 7 #MOVES
ROUND_END_BUFFER = 0.01 # Seconds from end of round. Needed because simulate() doesn't keep track of time(only search depth) right now and mcts() may go over time as a result

import traceback

class SplendorState(GameState):
    def __init__(self, num_agents):
        pass
    
    class BoardState:
        def __init__(self, num_agents):
            pass
            
        def dealt_list(self):
            return [card for deck in self.dealt for card in deck if card]
            
    class AgentState:
        def __init__(self, _id):
            self.id = _id

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

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.init = True
        self.player = None
    
    def SelectAction(self,actions,game_state):
        STARTUP_TIME = 0
        if self.init:
            # STARTUP_TIME = 4
            mdp = Splendor(self.id, len(game_state.agents), game_state)
            self.player = MCTS(mdp)
            self.init = False
        # Get best action string
        action = None
        try:
            # profiler = cProfile.Profile()
            # profiler.enable()
            root = self.player.mcts(game_state, ROUND_TIME+STARTUP_TIME-ROUND_END_BUFFER)
            action = root.getBestAction()
            # self.player.root = root.children[action].children[0]
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('tottime')
            # stats.print_stats()
        except Exception as e:
            traceback.print_exc()
            pass
        STARTUP_TIME = 0
        match = [x for x in actions if str(x)==action]
        return match[0] if match else actions[0]



'''
CODE FOR MONTE-CARLO TREE SEARCH

Code bases:
Adapted code from: https://gibberblot.github.io/rl-notes/single-agent/MDPs.html
Adapted code from: https://gibberblot.github.io/rl-notes/single-agent/mcts.html 

Sections: 
1. MDP class
2. MDP implementation for Splendor
3. MultiArmedBandits classes
3. Node classes
4. Monte-Carlo class

'''

'''
1. MDP abstraction
'''
class MDP:
    ''' Return all states of this MDP '''
    def getStates(self): abstract

    ''' Return all actions with non-zero probability from this state '''
    def getActions(self, state): abstract

    ''' Return all non-zero probability transitions for this action from this state '''
    def getTransitions(self, state, action): abstract

    ''' Return the reward for transitioning from state to nextState via action '''
    def getReward(self, state, action, nextState): abstract

    ''' Return the discount factor for this MDP '''
    def getDiscountFactor(self): abstract

    ''' Return the initial state of this MDP '''
    def getInitialState(self): abstract

    ''' Return all goal states of this MDP '''
    def getGoalStates(self): abstract

'''
2.Splendor MODEL. Use the splendor gamestate
CURRENT TODO 
    1. FILL IN THE FUNCTIONS BELOW
    
NEXT TODO:
    1. ???
'''
def equalStates(state1:SplendorState, state2:SplendorState):
    aid1, aid2 = state1.agent_to_move, state2.agent_to_move
    agent1, agent2 = state1.agents[aid1], state2.agents[aid2]
    if aid1!=aid2: return False #Check same agent to move 
    if state1.agents[aid1].last_action != state2.agents[aid2].last_action: return False #Check last actions equal
    for colour in agent1.gems: #Check gems are the same
        if agent1.gems[colour]!=agent2.gems[colour]:
            return False
    if str(state1.board.dealt) != str(state2.board.dealt): return False #Check cards dealt are the same
    return True
    


class Splendor(MDP):

    #CONSTANTS
    NUM_PLAYERS = 2
    DISCOUNT_FACTOR = 0.9


    '''Construct with rules'''
    def __init__(self, id, num_players, initialState:SplendorState):
        self.initialState = copy(initialState)
        self.rules = SplendorGameRule2(num_players)
        self.id = id
        self.num_players = num_players
        self.oid = (id+1)%num_players # opponent id


    ''' Return all states of this MDP '''
    def getStates(self): pass

    ''' Return all actions with non-zero probability from this state '''
    def getActions(self, state): 
        return self.rules.getLegalActions(state, state.agent_to_move)

    ''' Return all non-zero probability transitions for this action from this state '''
    def getTransitions(self, state, action):  # Jesus
        if action['type']=='pass':
            return [(state,1)]
        nextState = copy(state)
        aid = state.agent_to_move
        nextState.agent_to_move = (aid+1)%self.num_players
        nextState.agents[aid].last_action = action

        if action['noble']:
            for i in range(len(state.board.nobles)):
                if state.board.nobles[i][0] == action['noble'][0]:
                    del nextState.board.nobles[i]
                    nextState.agents[aid].nobles.append(action['noble'])
                    nextState.agents[aid].score += 3

        # both buy actions do not involve collecting gems only returning them
        if 'collect' in action['type'] or action['type'] == 'reserve':
            for colour, count in action['collected_gems'].items():
                nextState.board.gems[colour] -= count
                nextState.agents[aid].gems[colour] += count
            for colour, count in action['returned_gems'].items():
                nextState.agents[aid].gems[colour] -= count
                nextState.board.gems[colour] += count

        # if only collecting gems the state has no more changes, otherwise update card locations
        if 'collect' in action['type']:
            return [(nextState, 1)]
        elif 'card' in action:
            boughtCard = action['card']
            # if buying a reserved card theres no need to restock the board
            if 'buy' in action['type']:
                for colour, count in action['returned_gems'].items():
                    nextState.agents[aid].gems[colour] -= count
                    nextState.board.gems[colour] += count
                if action['type'] == 'buy_reserve':
                    for i in range(len(state.agents[aid].cards['yellow'])):
                        if state.agents[aid].cards['yellow'][i].code == boughtCard.code:
                            nextState.agents[aid].cards[boughtCard.colour].append(boughtCard)
                            nextState.agents[aid].score += boughtCard.points
                            del nextState.agents[aid].cards['yellow'][i]
                    return [(nextState, 1)]
                if action['type'] == 'buy_available':
                    nextState.agents[aid].cards[boughtCard.colour].append(boughtCard)
                    nextState.agents[aid].score += boughtCard.points

            # add card to respective pile
            elif action['type'] == 'reserve':
                nextState.agents[aid].cards['yellow'].append(boughtCard)

            # Remove card from list of rows of dealt cards
            nextState.board.dealt[boughtCard.deck_id].remove(boughtCard)

            return [(nextState, 1)]


    ''' Pick a transition generated by state and action '''
    def execute(self, state, action):
        nextState, distr = self.getTransitions(state, action)[0]
        return (nextState, self.getReward(state, action, nextState))


    ''' Return the reward for transitioning from state to nextState via action '''
    def getReward(self, state, action, nextState):  # Allison
        # 'collect' :           += # gems
        #  buy card :           card points + 4 (since developments>gems)
        # 'noble' :             6 points per noble
        
        aid = state.agent_to_move
        agent = state.agents[aid]
        opp = state.agents[(aid+1)%len(state.agents)]
        reward_point = agent.score - opp.score
        if nextState.agents[aid].score >= 15: # Winning is ultimate reward
            reward_point += 9999
        if 'buy' in action['type']:
            card = action['card']
            cost = sum(action['returned_gems'].values())
            reward_point += card.points/(cost+5) - len(agent.cards[card.colour])/(card.points+1)# Reward development based on point to cost ratio
            for noble in state.board.nobles: # More points if bought development relevant to nobles
                if cost>0 and card.colour in noble[1] and len(agent.cards[card.colour])<noble[1][card.colour]:
                    reward_point+= card.points/cost + 4
        if action['noble']: # Reward Noble
            reward_point += 6
            if agent.score + 3 >=15:
                reward_point += 9999
        if 'collected_gems' in action:
            if len(action['collected_gems'])>1:
                for colour in action['collected_gems']: # More points if get gems relevant to cards dealt
                    for row in nextState.board.dealt+[agent.cards['yellow']]:
                        for card in row:
                            if colour in card.cost and agent.gems[colour]+len(agent.cards[colour]) < card.cost[colour]: #Reward picking gems if it can be used next buy
                                cost = 0
                                for colour in card.cost:
                                    cost += max(0, card.cost[colour] - len(agent.cards[colour]))
                                reward_point += (card.points+0.3)/cost # Rewards for gems for each development with more reward if closer to buying
                                for noble in state.board.nobles: # More points if gems go to buying developments relevant to nobles
                                    if card.colour in noble[1] and len(agent.cards[card.colour])<noble[1][card.colour]:
                                        reward_point+= (card.points+2)/cost
            else:
                reward_point -= 0.5 * len(action['returned_gems']) # Less rewards for returning gems
        
        if action['type']=="reserve": #Reward for reserving cards. 
            opponent = state.agents[(aid+1)%len(state.agents)]
            card = action['card']
            gems1 = gems2 = 0
            for colour in card.cost:
                gems1 += max(0, card.cost[colour] - len(agent.cards[colour]) - agent.gems[colour])
                gems2 += max(0, card.cost[colour] - len(opponent.cards[colour]) - opponent.gems[colour])
            reward_point+= (card.points)/(gems2+8) + (card.points)/(gems1+8)
        
        # reward_point /= (sum([len(cards) for cards in agent.cards.values()])+1.0) # Divide reward by number of cards. states with less cards get higher rewards.

        return reward_point if self.id==aid else -reward_point

        
    ''' Return the discount factor for this MDP '''
    def getDiscountFactor(self): 
        return self.DISCOUNT_FACTOR

    ''' Return the initial state of this MDP '''
    def getInitialState(self):
        return self.initialState

    ''' Return all goal states of this MDP '''
    def getGoalStates(self): pass

    ''' Is state end of the game? '''
    def isTerminal(self, state):
        for agent in state.agents:
            if agent.score >= 15:
                return True
        return False


'''
3. Multi-Armed Bandits for selecting actions
'''
import random
import math

class MultiArmedBandit():

    '''
        Select an action given Q-values for each action.
    '''
    def select(self, actions, qValues): abstract
    
    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0  #number of times a choice has been made
        self.N = dict() #number of times each action has been chosen

    def select(self, actions, qValues):

        # First execute each action one time
        for action in actions:
            if not action in self.N.keys():
                self.N[action] = 1
                self.total += 1
                return action

        maxActions = []
        maxValue = float('-inf')
        for action in actions:
            N = self.N[action]
            value = qValues[action] + 2*math.sqrt((2 * math.log(self.total)) / (2*N))
            if value > maxValue:
                maxActions = [action]
                maxValue = value
            elif value == maxValue:
                maxActions += [action]
                    
        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(maxActions)
        self.N[result] = self.N[result] + 1
        self.total += 1
        return result


'''
4. Nodes for Expectimax trees in MCTS
'''    
import math
import random
import time
# from multi_armed_bandits import *

class Node():    

    # record a unique node id to distinguish duplicated states for visualisation
    nextNodeID = 0
    
    def __init__(self, mdp, parent, state):
        self.mdp = mdp  
        self.parent = parent
        self.state = state
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0

    '''
    Return the value of this node
    '''
    def getValue(self):
        return self.value

class StateNode(Node):
    
    def __init__(self, mdp, parent, state, reward = 0, probability = 1.0, bandit = UpperConfidenceBounds()):
        super().__init__(mdp, parent, state)
        
        # a dictionary from actions to an environment node
        self.children = {}

        # the reward received for this state
        self.reward = reward
        
        # the probability of this node being chosen from its parent
        self.probability = probability

        # a multi-armed bandit for this node
        self.bandit = bandit

    '''
    Return true if and only if all child actions have been expanded
    '''
    def isFullyExpanded(self):
        validActions = self.mdp.getActions(self.state)
        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self):
        if not self.isFullyExpanded():
            return self
        else:
            bestAction = self.getBestAction()
            return self.children[bestAction].select()

    def getBestAction(self):
        actions = list(self.children.keys())    
        qValues = dict()
        for action in actions:
            #get the Q values from all outcome nodes
            qValues[action] = self.children[action].getValue()
        return self.bandit.select(actions, qValues)

    def expand(self):
        #randomly select an unexpanded action to expand
        # New solution: Convert to strings as above, then retrieve which corresponding actions are unexpanded
        actions = self.mdp.getActions(self.state)
        action_strings = map(str,actions)
        # Set difference between expanded actions and actions = unexpanded actions
        unexpanded_action_strings = action_strings - self.children.keys() #NOTE: it's something about dictionary keys needing to be immutable. action dictionaries are mutable
        # Retrieve corresponding action dictionaries
        actions = [action for action in actions if str(action) in unexpanded_action_strings]
        action = random.choice(list(actions))
        #choose an outcome
        newChild = EnvironmentNode(self.mdp, self, self.state, action)
        newStateNode = newChild.expand()
        self.children[str(action)] = newChild
        return newStateNode

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits) 
        
        if self.parent != None:
            self.parent.backPropagate(reward)

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = round(self.children[action].getValue(), 3)
        return qValues

class EnvironmentNode(Node):
    
    def __init__(self, mdp, parent, state, action):
        super().__init__(mdp, parent, state)
        self.outcmes = {}
        self.action = action
        
        # a set of outcomes
        self.children = []

    def select(self):
        # choose one outcome based on transition probabilities. Problem reduced to no probabilistic transitions
        # (newState, reward) = self.mdp.execute(self.state, self.action)

        #find the corresponding state
        for child in self.children:
            # if equalStates(newState,child.state):
            return child.select()  # no longer probabilistic so only one child

    def addChild(self, action, newState, reward, probability):
        child = StateNode(self.mdp, self, newState, reward, probability)
        self.children += [child]
        return child

    def expand(self):
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)
        # expand all outcomes
        selected = None
        transitions = self.mdp.getTransitions(self.state, self.action)
        for (outcome, probability) in transitions:
            newChild = self.addChild(self.action, outcome, reward, probability)
            # find the child node correponding to the new state
            if equalStates(outcome,newState): 
                selected = newChild
        return selected

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((reward - self.value) / self.visits)
        self.parent.backPropagate(reward * self.mdp.getDiscountFactor())


'''
5. MCTS class 
'''
class MCTS():

    def __init__(self, mdp):
        self.mdp = mdp
        self.root = None

    '''
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    '''
    def mcts(self, currentState, timeout):
        startTime = int(time.time() * 1000)
        # Reused old tree if possible
        # if self.root==None:
        rootNode = StateNode(self.mdp, None, currentState)
        # else:
        #     last_action = str(currentState.agents[self.mdp.oid].last_action)
        #     if last_action in self.root.children:
        #         child = self.root.children[last_action].children[0]
        #         rootNode = child if equalStates(child.state, currentState) else StateNode(self.mdp, None, currentState)
        #     else:
        #         rootNode =  StateNode(self.mdp, None, currentState)

        i=0
        currentTime = int(time.time() * 1000)
        while currentTime < startTime + timeout * 1000:
            # find a state node to expand
            selectedNode = rootNode.select()
            if not self.mdp.isTerminal(selectedNode.state):
                child = selectedNode.expand()
                reward = self.simulate(child)
                child.backPropagate(reward)
                
            currentTime = int(time.time() * 1000)
            i+=1
        # print("myTeam.py iterations:", i)
        return rootNode

    '''
        Choose a random action. Heustics can be used here to improve simulations.
    '''
    def choose(self, state):
        actions = self.mdp.getActions(state)
        heuristics = np.array([self.heuristic(action, state) for action in actions])
        if sum(heuristics)==0: return random.choice(actions)
        heuristics = heuristics/sum(heuristics)
        return np.random.choice(actions, p=heuristics)
    
    def heuristic(self, action, state):
        aid = state.agent_to_move
        agent = state.agents[aid]
        opp = state.agents[(aid+1)%len(state.agents)]
        noble = 0
        if action['noble']:
            noble = 3
        if agent.score+noble >= 15: return 50
        value= agent.score-opp.score+20+noble
        if 'buy' in action['type']:
            card = action['card']
            if agent.score+card.points+noble>=15: return 50
            cost = len(action['returned_gems'])
            value += (card.points+1)/(cost+1)
            for _, noble in state.board.nobles:
                if card.colour in noble and len(agent.cards[card.colour])<noble[card.colour]:
                    value += 1/(cost+1)
        if action['type']=='reserve':
            card = action['card']
            cost = sum([max(0,card.cost[colour]-len(agent.cards[colour])) for colour in card.cost])
            value+= (card.points)/(cost+2)
        if 'collected_gems' in action:
            value += 1
            
        return min(29, value)

    '''
        Simulate until a terminal state
    '''
    def simulate(self, node):
        state = node.state
        cumulativeReward = 0.0
        depth = 0
        while depth<SEARCH_DEPTH: #NOTE: SAME AS BEFORE. STOP EARLY
            #choose an action to execute
            action = self.choose(state)
            
            # execute the action
            (newState, reward) = self.mdp.execute(state, action)
            #print("Reward:", reward)
            # discount the reward 
            cumulativeReward += pow(self.mdp.getDiscountFactor(), depth) * reward
            depth += 1

            state = newState
            
        return cumulativeReward

class SplendorGameRule2(SplendorGameRule):

    #Generate a list of gem combinations that can be returned, if agent exceeds limit with collected gems.
    #Agents are disallowed from returning gems of the same colour as those they've just picked up. Since collected_gems
    #is sampled exhaustively, this function simply needs to screen out colours in collected_gems, in order for agents
    #to be given all collected/returned combinations permissible.
    def generate_return_combos(self, current_gems, collected_gems):
        total_gem_count = sum(current_gems.values()) + sum(collected_gems.values())
        if total_gem_count > 10:
            return_combos = []
            num_return = total_gem_count - 10
            #Combine current and collected gems. Screen out gem colours that were just collected.
            total_gems = {i: current_gems.get(i, 0) + collected_gems.get(i, 0) for i in set(current_gems)}
            total_gems = {i[0]:i[1] for i in total_gems.items() if i[0] not in collected_gems.keys()}.items()
            #Form a total gems list (with elements == gem colours, and len == number of gems).
            total_gems_list = []                    
            for colour,count in total_gems:
                for _ in range(count):
                    total_gems_list.append(colour)
            #If, after screening, there aren't enough gems that can be returned, return an empty list, indicating that 
            #the collected_gems combination is not viable.
            if len(total_gems_list) < num_return:
                return []     
            #Else, find all valid combinations of gems to return.               
            for combo in set(itertools.combinations(total_gems_list, num_return)):
                returned_gems = {c:0 for c in COLOURS.values()}
                for colour in combo:
                    returned_gems[colour] += 1
                #Filter out colours with zero gems, and append.
                return_combos.append(dict({i for i in returned_gems.items() if i[-1]>0}))
                
            return return_combos
        
        return [{}] #If no gems need to be returned, return a list comprised of one empty combo.

    def getLegalActions(self, game_state, agent_id):
            actions = []
            agent,board = game_state.agents[agent_id], game_state.board
            
            #A given turn consists of the following:
            #  1. Collect gems (up to 3 different)    OR
            #     Collect gems (2 same, if stack >=4) OR
            #     Reserve one of 12 available cards   OR
            #     Buy one of 12 available cards       OR
            #     Buy a previously reserved card.
            #  2. Discard down to 10 gems if necessary.
            #  3. Obtain a noble if requirements are met.
            
            #Since the gamestate does not change during an agent's turn, all turn parts are able to be planned for at once.
            #Action fields: {'type', 'collected_gems', 'returned_gems', 'card', 'noble'}
            
            #Actions will always take the form of one of the following three templates:
            # {'type': 'collect_diff'/'collect_same', 'collected_gems': {gem counts}, 'returned_gems': {gem counts}, 'noble': noble}
            # {'type': 'reserve', 'card':card, 'collected_gems': {'yellow': 1/None}, 'returned_gems': {colour: 1/None}, 'noble': noble}
            # {'type': 'buy_available'/'buy_reserve', 'card': card, 'returned_gems': {gem counts}, 'noble': noble}
            
            #First, check if any nobles are waiting to visit from the last turn. Ensure each action to follow recognises
            #this, and in the exceedingly rare case that there are multiple nobles waiting (meaning that, at the last turn,
            #this agent had the choice of at least 3 nobles), multiply all generated actions by these nobles to allow the
            #agent to choose again.
            potential_nobles = []
            for noble in board.nobles:
                if self.noble_visit(agent, noble):
                    potential_nobles.append(noble)
            if len(potential_nobles) == 0:
                potential_nobles = [None]
            
            can_take = 10 - sum(agent.gems.values())
            #Generate actions (collect up to 3 different gems). Work out all legal combinations. Theoretical max is 10.
            available_colours = [colour for colour,number in board.gems.items() if colour!='yellow' and number>0 and can_take>1]
            for combo_length in range(3, min(len(available_colours), 3) + 1):
                for combo in itertools.combinations(available_colours, combo_length):
                    collected_gems = {colour:1 for colour in combo}
                    
                    #Find combos of gems to return, if any. Since the max to be returned can be 3, theoretical max 
                    #combinations will be 51, and max actions generated by the end of this stage will be 510. 
                    #Handling this branching factor properly will be crucial for agent performance.
                    #If return_combos comes back False, then taking these gems is invalid and won't be added.
                    return_combos = self.generate_return_combos(agent.gems, collected_gems)
                    for returned_gems in return_combos:
                        if sum(returned_gems.values()) >1: continue
                        for noble in potential_nobles:
                            actions.append({'type': 'collect_diff',
                                            'collected_gems': collected_gems,
                                            'returned_gems': returned_gems,
                                            'noble': noble})
            
            #Generate actions (collect 2 identical gems). Theoretical max is 5.
            available_colours = [colour for colour,number in board.gems.items() if colour!='yellow' and number>=4 and can_take>0]
            for colour in available_colours:
                collected_gems = {colour:2}
                
                #Like before, find combos to return, if any. Since the max to be returned is now 2, theoretical max 
                #combinations will be 21, and max actions generated here will be 105.
                return_combos = self.generate_return_combos(agent.gems, collected_gems)
                for returned_gems in return_combos:
                    if sum(returned_gems.values()) >1: continue
                    for noble in potential_nobles:
                        actions.append({'type': 'collect_same',
                                        'collected_gems': collected_gems,
                                        'returned_gems': returned_gems,
                                        'noble': noble})  

            
                
            #Generate actions (buy card). Agents can buy cards if they can cover its resource cost. Resources can come from
            #an agent's gem and card stacks. Card stacks represent gem factories, or 'permanent gems'; if there are 2 blue 
            #cards already purchased, this acts like 2 extra blue gems to spend in a given turn. Gems are therefore only 
            #returned if the stack of that colour is insufficient to cover the cost.
            #Agents are disallowed from purchasing > 7 cards of any one colour, for the purposes of a clean interface. 
            #This is not expected to affect gameplay, as there is essentially zero strategic reason to exceed this limit.
            #Available cards consist of cards dealt onto the board, as well as cards previously reserved by this agent.
            #There is a max 15 actions that can be generated here (15 possible cards to be bought: 12 dealt + 3 reserved).
            #However, in the case that multiple nobles are made candidates for visiting with this move, this number will
            #be multiplied accordingly. This however, is a rare event.
            for card in board.dealt_list() + agent.cards['yellow']:
                if not card or len(agent.cards[card.colour]) == 7:
                    continue    
                returned_gems = self.resources_sufficient(agent, card.cost) #Check if this card is affordable.
                if type(returned_gems)==dict: #If a dict was returned, this means the agent possesses sufficient resources.
                    #Check to see if the acquisition of a new card has meant new nobles becoming candidates to visit.
                    new_nobles = []
                    for noble in board.nobles:
                        agent_post_action = agent
                        #Give the card featured in this action to a copy of the agent.
                        agent_post_action.cards[card.colour].append(card)
                        #Use this copied agent to check whether this noble can visit.
                        if self.noble_visit(agent_post_action, noble):
                            new_nobles.append(noble) #If so, add noble to the new list.
                        agent.cards[card.colour].remove(card)
                    if not new_nobles:
                        new_nobles = [None]
                    for noble in new_nobles:
                        actions.append({'type': 'buy_reserve' if card in agent.cards['yellow'] else 'buy_available',
                                        'card': card,
                                        'returned_gems': returned_gems,
                                        'noble': noble})
                                        
            #Generate actions (reserve card). Agent can reserve only if it possesses < 3 cards currently reserved.
            #With a reservation, the agent will receive one seal (yellow), if there are any left. Reservations are stored
            #and displayed under the agent's yellow stack, as they won't generate their true colour until fully purchased.
            #There is a possible 12 cards to be reserved, and if the agent goes over limit, there are max 6 gem colours
            #that can be returned, leading to a theoretical max of 72 actions here.
            opp = game_state.agents[(agent_id+1)%len(game_state.agents)]
            if len(agent.cards['yellow']) < 3:
                collected_gems = {'yellow':1} if board.gems['yellow']>0 else {}
                return_combos = self.generate_return_combos(agent.gems, collected_gems)
                for returned_gems in return_combos:
                    for card in board.dealt_list():
                        if card and (card.points+1)/(sum(card.cost.values()))>0.24:
                            c1 = c2 = 0
                            for colour in card.cost:
                                c1 += max(0, card.cost[colour]-len(agent.cards[colour])-agent.gems[colour])
                                c2 += max(0, card.cost[colour]-len(opp.cards[colour])-opp.gems[colour])
                            deck = card.deck_id
                            if c1 == 0 or (c1>2 and c2>2) or deck<1: #(opp.score>0 and agent.score>0 and deck==0) or (opp.score==0 and agent.score==0 and deck>1): 
                                if len(actions)>0: continue
                            for noble in potential_nobles:
                                actions.append({'type': 'reserve',
                                                'card': card,
                                                'collected_gems': collected_gems,
                                                'returned_gems': returned_gems,
                                                'noble': noble})

            #Return list of actions. If there are no actions (almost impossible), all this player can do is pass.
            #A noble is still permitted to visit if conditions are met.
            if not actions:
                for noble in potential_nobles:
                    actions.append({'type': 'pass', 'noble':noble})
                    
            return actions
