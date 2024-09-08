from itertools import cycle, combinations, combinations_with_replacement
from typing import Dict, Literal, List, Tuple, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass
from collections import Counter

import gymnasium as gym

from Engine.template import Agent
from Engine.Splendor.splendor_model import SplendorState, SplendorGameRule
from Engine.Splendor import splendor_utils


COLORS = [c for c in splendor_utils.COLOURS.values() if c != "yellow"]
# ColorType = Literal[*COLORS]


class ActionType(Enum):
    PASS = auto()
    COLLECT_SAME = auto()
    COLLECT_DIFF = auto()
    RESERVE = auto()
    BUY_AVAILABLE = auto()
    BUY_RESERVED = auto()


@dataclass
class CardPosition:
    row: int
    column: int
    reserved_index: int


@dataclass
class Action:
    type: ActionType
    collected_gems: Optional[Dict[str, int]] = None
    returned_gems: Optional[Dict[str, int]] = None
    position: Optional[CardPosition] = None
    noble: Optional[Tuple[str, Dict[str, int]]] = None


def card_gen():
    for deck_num in range(3):
        for card_num in range(4):
            yield CardPosition(deck_num, card_num, -1)


def generate_all_collect_same_actions():
    for c in COLORS:
        for noble in [None] + splendor_utils.NOBLES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_SAME,
                noble=noble,
                collected_gems={c: 2},
            )
            # return 2 gems of different colors.
            for c1, c2 in combinations(filter(lambda x: x != c, COLORS), 2):
                yield Action(
                    type=ActionType.COLLECT_SAME,
                    noble=noble,
                    collected_gems={c: 2},
                    returned_gems={c1: 1, c2: 1},
                )
            # return gems of the same color.
            for num_gems_to_return in [1, 2]:
                for c_other in filter(lambda x: x != c, COLORS):
                    yield Action(
                        type=ActionType.COLLECT_SAME,
                        noble=noble,
                        collected_gems={c: 2},
                        returned_gems={c_other: num_gems_to_return},
                    )


def generate_all_collect_different_actions():
    for c1, c2, c3 in combinations(COLORS, 3):
        for noble in [None] + splendor_utils.NOBLES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_DIFF,
                noble=noble,
                collected_gems={c1: 1, c2: 1, c3: 1},
            )
            for num_gems_to_return in [1, 2, 3]:
                for to_return in combinations_with_replacement(
                    filter(lambda x: x not in [c1, c2, c3], COLORS)
                ):
                    yield Action(
                        type=ActionType.COLLECT_DIFF,
                        noble=noble,
                        collected_gems={c1: 1, c2: 1, c3: 1},
                        returned_gems=dict(Counter(to_return)),
                    )


def generate_all_buy_reserved_card_actions():
    for reserved_index in range(3):
        for noble in [None] + splendor_utils.NOBLES:
            yield Action(
                type=ActionType.BUY_AVAILABLE,
                noble=noble,
                position=CardPosition(-1, -1, reserved_index),
            )


def generate_all_buy_available_card_actions():
    for position in card_gen():
        for noble in [None] + splendor_utils.NOBLES:
            yield Action(type=ActionType.BUY_AVAILABLE, noble=noble, position=position)


ALL_ACTIONS = [
    # Do nothing.
    Action(type=ActionType.PASS),
    # Only acquire a noble.
    *[Action(type=ActionType.PASS, noble=noble) for noble in splendor_utils.NOBLES],
    # Reserve a card and get a yellow gem.
    *[
        Action(
            type=ActionType.RESERVE,
            collected_gems={"yellow": 1},
            position=position,
        )
        for position in card_gen()
    ],
    # Reserve a card.
    *[
        Action(
            type=ActionType.RESERVE,
            collected_gems={},
            position=position,
        )
        for position in card_gen()
    ],
    # Collect 2 stones of the same color.
    *list(generate_all_collect_same_actions()),
    # Collect 3 stones of different colors.
    *list(generate_all_collect_different_actions()),
    # Buy a Reserved Card.
    *list(generate_all_buy_reserved_card_actions()),
    # Buy an Available Card.
    *list(generate_all_buy_available_card_actions()),
]


class SplendorEnv(gym.Env):
    def __init__(self, agents: List[Agent], *args, **kwargs):
        """
        Create a new environment.
        """
        self.agents = agents
        self.number_of_players = len(self.agents)
        self.game_rule = SplendorGameRule(self.number_of_players)

        self.action_space = gym.spaces.Discrete(len(ALL_ACTIONS))
        self.observation_space = ...

        self.state = self.reset()

    def reset(self):
        """
        Reset the environment - Create a new game.
        """
        state = SplendorState(self.number_of_players)
        self.turns_gen = cycle(range(self.number_of_players))
        self.state = state
        return self.state

    def step(self, action: Action) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Run one timestep of the environment’s dynamics.

        :param: which action to take.
        :return: The new state (successor), the reward given, a flag indicating whether or
                 not the game ended, truncated (will be ignored), additional
                 information (will be ignored).

        :note: this method returns 2 redundant variables (truncated & info) only in
               order to comply with gym.Env.step signature.
        """
        turn = next(self.turns_gen)
        previous_score = self.state.agents[turn].score
        action_to_take = build_action(action)

        # generateSuccessor return a reference to the same
        # state object which is updated in-place.
        next_state = self.game_rule.generateSuccessor(self.state, action_to_take, turn)
        self.state = next_state
        current_score = self.state.agents[turn].score

        if current_score >= 15:
            reward = 100
        else:
            reward = current_score - previous_score

        # this refers to the next_state.
        terminated = self.game_rule.gameEnds()

        return (next_state, reward, terminated, False, {})

    def render(self):
        # Don't render anything.
        pass
