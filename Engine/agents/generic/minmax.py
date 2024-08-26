import numpy as np
from template import Agent
from copy import copy

DEPTH = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state, game_rule):
        return self.SelectActionRecursion(game_state, game_rule, DEPTH)[0]

    def SelectActionRecursion(self, game_state, game_rule, depth,
                              is_maximizing=True, alpha=-np.inf, beta=np.inf):
        agent_id = int(not is_maximizing)
        actions = game_rule.getLegalActions(game_state, agent_id)
        if depth == 0 or len(actions) == 0:
            return None, self.evaluation_function(game_state, agent_id)

        agent, board = (copy(game_state.agents[agent_id]),
                        copy(game_state.board))

        if is_maximizing:
            best_value = -np.inf
            best_action = None
            for action in actions:
                next_state = game_rule.generateSuccessor(game_state,
                                                         action, agent_id)
                game_state.agents[agent_id] = agent
                game_state.board = board
                _, action_value = self.SelectActionRecursion(next_state,
                                                              game_rule, depth - 1,
                                                             not is_maximizing, alpha, beta)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action
                if best_value >= beta:
                    break

        else:
            best_value = np.inf
            best_action = None
            for action in actions:
                next_state = game_rule.generateSuccessor(game_state,
                                                         action, agent_id)
                game_state.agents[agent_id] = agent
                game_state.board = board
                _, action_value = self.SelectActionRecursion(next_state,
                                                             game_rule, depth - 1,
                                                        not is_maximizing, alpha, beta)
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
                if best_value <= alpha:
                    break

        return best_action, best_value

    def evaluation_function(self, state, agent_id):
        agent_state = state.agents[agent_id]
        score_factor = 1
        cards_factor = 0.33
        gems_factor = 0.15
        return (agent_state.score * score_factor + len(agent_state.cards) *
                cards_factor + len(agent_state.gems) * gems_factor)