"""
Implementation of an agent that selects the first legal action.
"""

import operator
import random
from typing import override

import numpy as np

from splendor.splendor.constants import WINNING_SCORE_TRESHOLD
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

# This agent supports only a game of 2 players
EXPECTED_AMOUNT_OF_PLAYERS = 2
DEPTH = 2
GEMS_AMOUNT_THRESHOLD = 8


class MiniMaxAgent(Agent):
    """
    A Minimax agent, utilizing the zero-sum property of the game,
    there is only a single winner in each game, for determining which
    action to play.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        assert len(game_state.agents) == EXPECTED_AMOUNT_OF_PLAYERS
        selected_action = self._select_action_recursion(game_state, game_rule, DEPTH)[0]
        assert selected_action is not None
        return selected_action

    def _select_action_recursion(  # noqa: PLR0913,PLR0917
        self,
        game_state: SplendorState,
        game_rule: SplendorGameRule,
        depth: int,
        is_maximizing: bool = True,
        alpha: float = -np.inf,
        beta: float = np.inf,
    ) -> tuple[ActionType | None, float]:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        if depth == 0:
            return None, self._evaluation_function(game_state)
        agent_id = self.id if is_maximizing else 1 - self.id
        actions = game_rule.getLegalActions(game_state, agent_id)
        random.shuffle(actions)
        actions.sort(key=operator.itemgetter("type"))
        assert len(actions) != 0

        best_action = None
        best_value = -np.inf if is_maximizing else np.inf
        for action in actions:
            next_state = game_rule.generateSuccessor(game_state, action, agent_id)
            _, action_value = self._select_action_recursion(
                next_state, game_rule, depth - 1, not is_maximizing, alpha, beta
            )
            # generateSuccessor alternates the game_state inplace,
            # that's why we need to call generatePredecessor to revert
            # it (even though we do not need its output)
            _ = game_rule.generatePredecessor(game_state, action, agent_id)

            if is_maximizing:
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_action, best_value

    def _evaluation_function(self, state: SplendorState) -> float:
        # pylint: disable=too-many-locals
        agent_state = state.agents[self.id]
        score_factor = 2
        cards_factor = 0.7
        gems_factor = 0.1
        gems_var_factor = -0.2
        color_cost_factor = 0.1
        reward = 0

        max_score = max(agent.score for agent in state.agents)
        if max_score >= WINNING_SCORE_TRESHOLD:
            reward = 99999 + max_score
            if max_score > agent_state.score:
                reward *= -1
            return reward
        if sum(agent_state.gems.values()) >= GEMS_AMOUNT_THRESHOLD:
            gems_factor = -0.7
        gems_var = np.var(list(agent_state.gems.values()))

        for card in state.board.dealt_list() + agent_state.cards["yellow"]:
            relevant_to_nobles = 0
            for _, noble_cost in state.board.nobles:
                if (
                    card.colour in noble_cost
                    and len(agent_state.cards[card.colour]) < noble_cost[card.colour]
                ):
                    # Card is relevant for nobles, increase its weight
                    relevant_to_nobles += 1

            for color, color_cost in card.cost.items():
                reward -= abs(
                    (
                        color_cost
                        - (agent_state.gems[color] + len(agent_state.cards[color]))
                    )
                    * color_cost_factor
                    * (card.points + 1 + relevant_to_nobles * 0.5)
                )

        return (
            reward
            + agent_state.score * score_factor
            + len(agent_state.cards) * cards_factor
            + sum(agent_state.gems.values()) * gems_factor
            + gems_var * gems_var_factor
        )


myAgent = MiniMaxAgent  # pylint: disable=invalid-name
