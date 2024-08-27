import random
import numpy as np
from template import Agent
from copy import deepcopy
from timeit import default_timer as timer

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
        random.shuffle(actions)
        actions.sort(key=lambda action: action["type"])
        if depth == 0 or len(actions) == 0:
            return None, self.evaluation_function(game_state, agent_id)

        if is_maximizing:
            best_value = -np.inf
            best_action = None
            for action in actions:
                next_state = game_rule.generateSuccessor(game_state,
                                                         action, agent_id)
                _, action_value = self.SelectActionRecursion(next_state,
                                                              game_rule, depth - 1,
                                                             not is_maximizing, alpha, beta)
                # generateSuccessor alternates the game_state inplace,
                # that's why we need to call generatePredecessor to revert
                # it (even though we do not need its output)
                prev_state = game_rule.generatePredecessor(game_state,
                                                           action, agent_id)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action
                if best_value >= beta:
                    print("beta")
                    break

        else:
            best_value = np.inf
            best_action = None
            for action in actions:
                next_state = game_rule.generateSuccessor(game_state,
                                                         action, agent_id)
                _, action_value = self.SelectActionRecursion(next_state,
                                                             game_rule, depth - 1,
                                                        not is_maximizing, alpha, beta)
                # generateSuccessor alternates the game_state inplace,
                # that's why we need to call generatePredecessor to revert
                # it (even though we do not need its output)
                prev_state = game_rule.generatePredecessor(game_state,
                                                           action, agent_id)
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
                if best_value <= alpha:
                    print("alpha")
                    break

        return best_action, best_value

    @staticmethod
    def evaluation_function(state, agent_id):
        agent_state = state.agents[agent_id]
        score_factor = 2
        cards_factor = 0.7
        gems_factor = 0.1
        gems_var_factor = -0.2
        color_cost_factor = 0.1
        reward = 0

        if agent_state.score >= 15:
            return 99999
        if sum(agent_state.gems.values()) >= 8:
            gems_factor = -0.7
        gems_var = np.var(list(agent_state.gems.values()))

        for row in state.board.dealt + [agent_state.cards["yellow"]]:
            for card in [c for c in row if c]:  # filter Nones out
                relevant_to_nobles = 0
                for noble in state.board.nobles:
                    if (card.colour in noble[1] and len(agent_state.cards[
                                                            card.colour])
                            < noble[1][card.colour]):
                        # Card is relevant for nobles, increase its weight
                        relevant_to_nobles += 1

                for color, color_cost in card.cost.items():
                    reward -= ((color_cost - (agent_state.gems[color] +
                              len(agent_state.cards[color]))) * color_cost_factor *
                               (card.points + 1 + relevant_to_nobles * 0.5))

                # More points if cards are relevant to nobles
                # for noble in state.board.nobles:
                #     if (card.colour in noble[1] and len(agent_state.cards[
                #                                             card.colour])
                #             < noble[1][card.colour]):
                #         reward += * (card.points + 2) / cost
                    # if card.color in
                #
                #
                #     if
                # if (
                #         colour in card.cost
                #         and agent.gems[colour] + len(agent.cards[colour])
                #         < card.cost[colour]
                # ):  # Gems are close to cards on board
                #     cost = 0
                #     for cardColour in card.cost:
                #         cost += max(
                #             0,
                #             card.cost[cardColour]
                #             - len(agent.cards[cardColour]),
                #         )
                #     reward_point += count * (card.points + 0.3) / cost
                #     for (
                #             noble
                #     ) in (
                #             state.board.nobles
                #     ):  # More points if gems go to buying developments relevant to nobles
                #         if (
                #                 card.colour in noble[1]
                #                 and len(agent.cards[card.colour])
                #                 < noble[1][card.colour]
                #         ):
                #             reward_point += count * (card.points + 2) / cost

        return (reward + agent_state.score * score_factor + len(
            agent_state.cards) *
                cards_factor + sum(agent_state.gems.values()) * gems_factor
                + gems_var * gems_var_factor)
