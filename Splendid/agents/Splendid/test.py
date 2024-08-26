from typing import AsyncContextManager
from template import Action, Agent, GameState
from Splendor.splendor_utils import *
import random, itertools
from Splendor.splendor_model import SplendorGameRule, Card
import cProfile, pstats
import json

STARTUP_TIME = 15 #SECONDS
ROUND_TIME = 1 #SECOND
SEARCH_DEPTH = 3 #MOVES
ROUND_END_BUFFER = 0.05 # Seconds from end of round. Needed because simulate() doesn't keep track of time(only search depth) right now and mcts() may go over time as a result

import traceback

class SplendorState(GameState):           
    def __init__(self, num_agents):
        self.num_agents = num_agents
    
    class BoardState:
        def __init__(self, num_agents):
            self.num_agents = num_agents
            
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
        if self.init:
            mdp = Splendor(self.id, len(game_state.agents), game_state)
            self.player = MCTS(mdp)
            self.init = False
        # Get best action string
        action = None
        try:
            # profiler = cProfile.Profile()
            # profiler.enable()
            action = self.player.mcts(game_state, ROUND_TIME-ROUND_END_BUFFER).getBestAction()
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('tottime')
            # stats.print_stats()
        except Exception as e:
            traceback.print_exc()
        match = [x for x in actions if str(x)==action]
        #print("matching moves from mcts:",match)
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
    if aid1!=aid2: return False
    if state1.agents[aid1].last_action != state2.agents[aid2].last_action: return False
    if str(state1.board.dealt) != str(state2.board.dealt): return False
    # for row1, row2 in state1.board.dealt, state2.board.dealt:
    #     if len(row1)!=len(row2): return False
    #     for card1, card2 in row1,row2:
    #         if card1.code!=card2.code: return False
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
        # print("getting transitions", self.getTransitions(state, action))
        nextState, distr = self.getTransitions(state, action)[0]
        # print("gotten")
        # print("chosen")
        # print("reward:", self.getReward(state, action, nextState))
        return (nextState, self.getReward(state, action, nextState))


    ''' Return the reward for transitioning from state to nextState via action '''
    def getReward(self, state, action, nextState):  # Allison
        # 'collect' :           += # gems
        #  buy card :           card points + 4 (since developments>gems)
        # 'noble' :             6 points per noble
        reward_point = 0
        aid = state.agent_to_move
        if nextState.agents[aid].score >= 15:
            reward_point = 9999
        if 'buy' in action['type']:
            card = action['card']
            reward_point += 3*card.points + 8
        if action['noble']:
            reward_point += 8*len(action['noble'])
        if 'collected_gems' in action:
            reward_point += len(action['collected_gems'])
            if 'returned_gems' not in action and len(action['collected_gems'])>1:
                for colour in action['collected_gems']: # More points if get gems relevant to cards dealt
                    for row in nextState.board.dealt:
                        for card in row:
                            for noble in state.board.nobles: # More points if gems relevant tonobles
                                if card.colour in noble[1]:
                                    reward+= 1.0/15
                            if colour in card.cost:
                                reward_point += (card.points+1)/(15+max(0, card.cost[colour]-state.agents[aid].gems[colour]-len(state.agents[aid].cards[colour])))
            else:
                reward_point -= len(action['returned_gems'])
        
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

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''
    def runBandit(self, episodes = 2000, episodeLength = 1000, drift = True):

        #the actions available
        actions = [0, 1, 2, 3, 4]

        rewards = []
        for episode in range(0, episodes):
            self.reset()

            # The probability of receiving a payoff of 1 for each action
            probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]
        
            qValues = dict()
            N = dict()
            for action in actions:
                qValues[action] = 0.0
                N[action] = 0

            episodeRewards = []
            for step in range(0, episodeLength):

                # Halfway through the episode, change the probabilities
                if drift and step == episodeLength / 2:
                    probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]
                
                #select an action
                action = self.select(actions, qValues)

                r = random.random()
                reward = 0
                if r < probabilities[action]:
                    reward = 5

                episodeRewards += [reward]

                N[action] = N[action] + 1

                qValues[action] = qValues[action] - (qValues[action] / N[action])
                qValues[action] = qValues[action] + reward / N[action]

            rewards += [episodeRewards]

        return rewards

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
            value = qValues[action] + math.sqrt((2 * math.log(self.total)) / N)
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
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)

        #find the corresponding state
        for child in self.children:
            if equalStates(newState,child.state):
                return child.select()

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

    '''
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    '''
    def mcts(self, currentState, timeout):
        rootNode = StateNode(self.mdp, None, currentState)
        i=0
        startTime = int(time.time() * 1000)
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
        # print("iterations:", i)
        return rootNode

    '''
        Choose a random action. Heustics can be used here to improve simulations.
    '''
    def choose(self, state):
        return random.choice(self.mdp.getActions(state))

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
            
            #Generate actions (collect up to 3 different gems). Work out all legal combinations. Theoretical max is 10.
            available_colours = [colour for colour,number in board.gems.items() if colour!='yellow' and number>0]
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
            available_colours = [colour for colour,number in board.gems.items() if colour!='yellow' and number>=4]
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

            #Generate actions (reserve card). Agent can reserve only if it possesses < 3 cards currently reserved.
            #With a reservation, the agent will receive one seal (yellow), if there are any left. Reservations are stored
            #and displayed under the agent's yellow stack, as they won't generate their true colour until fully purchased.
            #There is a possible 12 cards to be reserved, and if the agent goes over limit, there are max 6 gem colours
            #that can be returned, leading to a theoretical max of 72 actions here.
            if len(agent.cards['yellow']) < 3 and len(agent.gems.values())>6:
                collected_gems = {'yellow':1} if board.gems['yellow']>0 else {}
                return_combos = self.generate_return_combos(agent.gems, collected_gems)
                for returned_gems in return_combos:
                    for card in board.dealt_list():
                        if card:
                            for noble in potential_nobles:
                                actions.append({'type': 'reserve',
                                                'card': card,
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
            
            #Return list of actions. If there are no actions (almost impossible), all this player can do is pass.
            #A noble is still permitted to visit if conditions are met.
            if not actions:
                for noble in potential_nobles:
                    actions.append({'type': 'pass', 'noble':noble})
                    
            return actions