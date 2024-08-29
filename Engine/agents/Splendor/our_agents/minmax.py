import random
import numpy as np
from template import Agent

# This agent supports only a game of 2 players

DEPTH = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state, game_rule):
        assert len(game_state.agents) == 2
        return self.select_action_recursion(game_state, game_rule, DEPTH)[0]

    def select_action_recursion(self, game_state, game_rule, depth,
                                is_maximizing=True, alpha=-np.inf, beta=np.inf):
        agent_id = self.id if is_maximizing else 1 - self.id
        actions = game_rule.getLegalActions(game_state, agent_id)
        random.shuffle(actions)
        actions.sort(key=lambda action: action["type"])
        assert len(actions) != 0
        if depth == 0:
            return None, self.evaluation_function(game_state)

        best_action = None
        best_value = -np.inf if is_maximizing else np.inf
        for action in actions:
            next_state = game_rule.generateSuccessor(game_state,
                                                     action, agent_id)
            _, action_value = self.select_action_recursion(next_state,
                                                           game_rule,
                                                           depth - 1,
                                                           not is_maximizing,
                                                           alpha, beta)
            # generateSuccessor alternates the game_state inplace,
            # that's why we need to call generatePredecessor to revert
            # it (even though we do not need its output)
            prev_state = game_rule.generatePredecessor(game_state,
                                                       action, agent_id)

            if is_maximizing:
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
                if best_value >= beta:
                    break
            else:
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
                if best_value <= alpha:
                    break

        return best_action, best_value

    def evaluation_function(self, state):
        agent_state = state.agents[self.id]
        score_factor = 2
        cards_factor = 0.7
        gems_factor = 0.1
        gems_var_factor = -0.2
        color_cost_factor = 0.1
        reward = 0

        if agent_state.score >= 15:
            return 99999 + agent_state.score
        if state.agents[1 - self.id].score >= 15:
            return -99999 - state.agents[1 - self.id].score
        if sum(agent_state.gems.values()) >= 8:
            gems_factor = -0.7
        gems_var = np.var(list(agent_state.gems.values()))

        for card in state.board.dealt_list() + agent_state.cards["yellow"]:
            relevant_to_nobles = 0
            for noble_id, noble_cost in state.board.nobles:
                if (card.colour in noble_cost and len(agent_state.cards[
                                                        card.colour])
                        < noble_cost[card.colour]):
                    # Card is relevant for nobles, increase its weight
                    relevant_to_nobles += 1

            for color, color_cost in card.cost.items():
                reward -= abs((color_cost - (agent_state.gems[
                                                       color] +
                          len(agent_state.cards[color]))) * color_cost_factor *
                           (card.points + 1 + relevant_to_nobles * 0.5))

        return (reward + agent_state.score * score_factor + len(
            agent_state.cards) *
                cards_factor + sum(agent_state.gems.values()) * gems_factor
                + gems_var * gems_var_factor)
