# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Lee Guo Yi, Gary Zhang, Rebonto Zaman Joarder
# Date:    30/9/2021
# Purpose: Implements A* Heuristic search, take best choice within time limit.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

from template import Agent
import time
from Splendor.splendor_model import *
import heapq

gemTypes = ["red", "green", "blue", "black", "white", "yellow"]
THINKTIME = 0.95
game_rule = SplendorGameRule(2)


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Priority Queue code taken from Assignment 1
class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def calculateHeuristicValue(nobles, each_action):
    buy_noble, points, gem_income, gem_card = action_changes(each_action)

    # score priority (how much points u give)
    """
    noble priority => give score if u can get a noble
    gem_card_priority => give score if u can get a gem card (game points dont matter)
    gem_income_priority => give score if you can get gems from the stashes
    points_priority => give score if you can get points
    """
    noble_priority = 100
    # gem card points is always 1 because it gets 1 gem card
    gem_card_priority = 10
    # gem income can be negative if buy a card. positive if collect gems or reserve.
    # reserve = 1, collect gems = 1 or 2 or 3
    gem_income_priority = 0.34
    # points priority can be 1 to 5
    points_priority = 20
    reserve_priority = 0.3

    # initiate the weights
    noble_weight = 0
    gem_card_weight = 0
    gem_income_weight = 0
    points_weight = 0
    reserve_weight = 0

    # calculate noble priority
    if buy_noble == True:
        noble_weight = 1
    else:
        noble_weight = 0

    nobles_combo = {}
    for noble in nobles:
        for gem, cost in noble.items():
            if gem in nobles_combo:
                nobles_combo[gem] = nobles_combo[gem] + cost
            else:
                nobles_combo[gem] = cost
    sorted_gem = sorted(nobles_combo, reverse=True, key=nobles_combo.get)

    max_nobles_combo = {}
    for noble in nobles:
        for gem, cost in noble.items():
            if gem in max_nobles_combo:
                if cost > max_nobles_combo[gem]:
                    max_nobles_combo[gem] = cost
            else:
                max_nobles_combo[gem] = cost

    # calculate gem_card weight
    gem_card_values = gem_card.values()
    gem_card_weight = sum(gem_card_values)

    # calculate gem income weight
    gem_income_values = list(gem_income.values())

    gem_income_weight_list = []
    for i in range(0, len(gem_income_values)):
        if gem_income_values[i] > 0:
            gem_income_weight_list.append(gem_income_values[i])
        else:
            gem_income_weight_list.append(0)

    gem_income_weight = sum(gem_income_weight_list)
    if gem_income_weight < 0:
        gem_income_weight = 0

    # calculate points weight
    points_weight = points

    # calculate reserve points
    if each_action["type"] == "buy_reserve":
        return 10000

    # sum them up and return the value
    noble_score = noble_weight * noble_priority
    gem_card_score = gem_card_weight * gem_card_priority
    gem_income_score = gem_income_weight * gem_income_priority
    points_score = points_weight * points_priority

    final_queue_score = gem_card_score + noble_score + gem_income_score + points_score
    if final_queue_score < 0:
        final_queue_score = 0

    # each step is 100 points, current actionable step is 1
    final_queue_score = 100 - final_queue_score

    if final_queue_score < 0:
        final_queue_score = 0

    return final_queue_score


class myAgent(Agent):
    # initialise the game board and state information
    def __init__(self, _id):
        super().__init__(_id)
        self.gr = SplendorGameRule(2)

    def SelectAction(self, actions, game_state):
        initiate_time = time.time()
        priorityQueue = PriorityQueue()
        init_game_state = self.gr.initialGameState

        # check current player buy ability
        own_buy_ability = currentAgentState(game_state, self.id)

        # get states of nobles
        nobles = available_nobles(game_state)

        for each_action in actions:
            current_time = time.time()
            time_taken = current_time - initiate_time
            if THINKTIME > time_taken:
                priorityQueue.push(
                    each_action,
                    self.a_star(
                        own_buy_ability, nobles, each_action, game_state, actions
                    ),
                )
            else:
                break

        return priorityQueue.pop()

    def agent_can_buy(self, cost, agent_gems):
        agent_gems_ret = agent_gems
        for key in cost:
            agent_gems_ret[key] = agent_gems_ret[key] - cost[key]

        for key in agent_gems_ret:
            if agent_gems_ret[key] > 0:
                agent_gems_ret[key] = 0
            else:
                agent_gems_ret[key] = abs(agent_gems_ret[key])

        return agent_gems_ret

    def a_star(self, own_buy_ability, nobles, each_action, game_state, actions):
        """
        a_star heuristic function
        own_buy_ablitiy how much resources you have to buy something
        """
        buy_noble, points, gem_income, gem_card = action_changes(each_action)
        new_heuristic = 0
        # plan to get the most points
        buy_now = calculateHeuristicValue(nobles, each_action)
        # buy_now refers to evaluating the current board and if can buy any cards.
        # heuristic value = 100 - any (point weights or gem cards). more than 90 means only can get gem resource.

        # has good points cards to buy this turn
        if buy_now <= 90:
            return calculateHeuristicValue(nobles, each_action)

        # no good cards to buy, looking at buying next turn
        elif buy_now > 90:
            successor_state = self.gr.generateSuccessor(
                game_state, each_action, self.id
            )
            # combine the decks together
            deck = (
                successor_state.board.dealt[0]
                + successor_state.board.dealt[1]
                + successor_state.board.dealt[2]
            )
            cards_can_buy = []
            for c in deck:
                if c is not None:
                    bool_suff = self.gr.resources_sufficient(
                        game_state.agents[self.id], c.cost
                    )

                    # when true
                    if bool_suff:
                        cards_can_buy.append(c)
                        return calculateHeuristicValue(nobles, each_action) - 5
                else:
                    continue

            # collect gems in according to deck, aim cheapest card and get gem resources
            if not cards_can_buy:
                final_gem_resource = []
                final_gem_resource_count = []
                for c in deck:
                    if c is not None:
                        ag = self.agent_can_buy(c.cost, copy.copy(own_buy_ability))
                        final_gem_resource.append(ag)
                        final_gem_resource_count.append(sum(ag.values()))
                    else:
                        continue

                smallest_index = final_gem_resource_count.index(
                    min(final_gem_resource_count)
                )
                card_to_take = final_gem_resource[smallest_index]

                weightage = 0
                for k in card_to_take:
                    if card_to_take[k] * gem_income[k] == 0:
                        weightage += 0.33
                    else:
                        weightage += card_to_take[k] * gem_income[k]
                if sum(gem_income.values()) < 3:
                    weightage = 0.2

                new_heuristic = 100 - (weightage * 0.34)
        return new_heuristic


# find the current state of the agent, gems he has etc buy ability
def currentAgentState(game_state, id):
    agent = game_state.agents[id]
    buy_ability = {}
    # print(game_state.agents[id].gems)
    for gem in gemTypes:
        buy_ability[gem] = agent.gems[gem] + len(agent.cards[gem])

    return buy_ability


def available_nobles(game_state):
    nobles = game_state.board.nobles
    potential_nobles = []
    for count, noble in enumerate(nobles):
        noble_gem_types = noble[1]
        potential_nobles.append(noble_gem_types)
    return potential_nobles


def action_changes(action):
    points = 0
    buy_noble = False
    gem_income = {"red": 0, "green": 0, "blue": 0, "black": 0, "white": 0, "yellow": 0}
    gem_card = {}
    if action["type"] == "collect_diff" or action["type"] == "collect_same":
        points = 0
        for gem in action["collected_gems"]:
            gem_income[gem] = action["collected_gems"][gem]

            if len(action["returned_gems"]) != 0:
                returned_gem_color = 0
                returned_gem_val = 0
                for key, val in action["returned_gems"].items():
                    returned_gem_color = key
                    returned_gem_val = val

                gem_income[returned_gem_color] = (
                    gem_income[returned_gem_color] - returned_gem_val
                )

    elif action["type"] == "buy_available" or action["type"] == "buy_reserve":
        points = action["card"].points
        for gem in action["returned_gems"]:
            returned_gem_color = 0
            returned_gem_val = 0
            for key, val in action["returned_gems"].items():
                returned_gem_color = key
                returned_gem_val = val
            gem_income[returned_gem_color] = (
                gem_income[returned_gem_color] - returned_gem_val
            )

        gem_card[action["card"].colour] = 1

    elif action["type"] == "reserve":
        gem_income["yellow"] = 1

    if action["noble"]:
        buy_noble = True
    return buy_noble, points, gem_income, gem_card


#
