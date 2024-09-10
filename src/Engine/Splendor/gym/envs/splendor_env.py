import random

from itertools import cycle, combinations, combinations_with_replacement
from typing import Dict, Literal, List, Tuple, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass
from collections import Counter

import gymnasium as gym

from Engine.template import Agent
from Engine.Splendor.splendor_model import SplendorState, SplendorGameRule
from Engine.Splendor import splendor_utils


NUMBER_OF_NOBLES = 3
NUMBER_OF_DECKS = 3
MAX_NUMBER_OF_CARDS_FROM_DECK = 4
MAX_NUMBER_OF_RESERVED_CARDS = 3
RESERVED = "yellow"
COLORS = [c for c in splendor_utils.COLOURS.values() if c != RESERVED]
NOBLES_INDICES = list(range(NUMBER_OF_NOBLES))
# ColorType = Literal[*COLORS]

GemsDict = Dict[str, int]


class ActionType(Enum):
    PASS = auto()
    COLLECT_SAME = auto()
    COLLECT_DIFF = auto()
    RESERVE = auto()
    BUY_AVAILABLE = auto()
    BUY_RESERVE = auto()


@dataclass
class CardPosition:
    tier: int
    card_index: int
    reserved_index: int


@dataclass
class Action:
    type: ActionType
    collected_gems: Optional[GemsDict] = None
    returned_gems: Optional[GemsDict] = None
    position: Optional[CardPosition] = None
    noble_index: Optional[int] = None


def card_gen():
    for deck_num in range(NUMBER_OF_DECKS):
        for card_num in range(MAX_NUMBER_OF_CARDS_FROM_DECK):
            yield CardPosition(deck_num, card_num, -1)


def generate_all_reserve_card_actions():
    for position in card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            # reserve a card, without collecting the yellow gem.
            yield Action(
                type=ActionType.RESERVE,
                collected_gems={},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            # reserve a card, collect a yellow gem
            yield Action(
                type=ActionType.RESERVE,
                collected_gems={"yellow": 1},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            for c in COLORS:
                # reserve a card, collect a yellow gem, and return a gem.
                yield Action(
                    type=ActionType.RESERVE,
                    collected_gems={"yellow": 1},
                    returned_gems={c: 1},
                    position=position,
                    noble_index=noble_index,
                )


def generate_all_collect_same_actions():
    for c in COLORS:
        for noble_index in [None] + NOBLES_INDICES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_SAME,
                noble_index=noble_index,
                collected_gems={c: 2},
                returned_gems={},
            )
            # return 2 gems of different colors.
            for c1, c2 in combinations(filter(lambda x: x != c, COLORS), 2):
                yield Action(
                    type=ActionType.COLLECT_SAME,
                    noble_index=noble_index,
                    collected_gems={c: 2},
                    returned_gems={c1: 1, c2: 1},
                )
            # return gems of the same color.
            for num_gems_to_return in [1, 2]:
                for c_other in filter(lambda x: x != c, COLORS):
                    yield Action(
                        type=ActionType.COLLECT_SAME,
                        noble_index=noble_index,
                        collected_gems={c: 2},
                        returned_gems={c_other: num_gems_to_return},
                    )


def generate_all_collect_different_actions():
    for c1, c2, c3 in combinations(COLORS, 3):
        for noble_index in [None] + NOBLES_INDICES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_DIFF,
                noble_index=noble_index,
                collected_gems={c1: 1, c2: 1, c3: 1},
                returned_gems={},
            )
            for num_gems_to_return in [1, 2, 3]:
                for to_return in combinations_with_replacement(
                    filter(lambda x: x not in [c1, c2, c3], COLORS),
                    num_gems_to_return,
                ):
                    yield Action(
                        type=ActionType.COLLECT_DIFF,
                        noble_index=noble_index,
                        collected_gems={c1: 1, c2: 1, c3: 1},
                        returned_gems=dict(Counter(to_return)),
                    )


def generate_all_buy_reserve_card_actions():
    for reserved_index in range(MAX_NUMBER_OF_RESERVED_CARDS):
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionType.BUY_AVAILABLE,
                noble_index=noble_index,
                position=CardPosition(-1, -1, reserved_index),
            )


def generate_all_buy_available_card_actions():
    for position in card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionType.BUY_AVAILABLE,
                noble_index=noble_index,
                position=position,
            )


ALL_ACTIONS = [
    # Do nothing.
    Action(type=ActionType.PASS),
    # Only acquire a noble.
    *[Action(type=ActionType.PASS, noble_index=noble) for noble in NOBLES_INDICES],
    # Reserve a card
    *list(generate_all_reserve_card_actions()),
    # Collect 2 stones of the same color.
    *list(generate_all_collect_same_actions()),
    # Collect 3 stones of different colors.
    *list(generate_all_collect_different_actions()),
    # Buy a Reserved Card.
    *list(generate_all_buy_reserve_card_actions()),
    # Buy an Available Card.
    *list(generate_all_buy_available_card_actions()),
]


class SplendorEnv(gym.Env):
    def __init__(self, agents: List[Agent], *args, **kwargs):
        """
        Create a new environment.

        :param agents: a list of all the opponents.
        """
        self.agents = agents
        self.number_of_players = len(self.agents) + 1
        self.game_rule = SplendorGameRule(self.number_of_players)

        self.action_space = gym.spaces.Discrete(len(ALL_ACTIONS))
        self.observation_space = gym.spaces.Discrete(1)

        self.state: SplendorState = self.reset()

    def reset(self) -> Tuple[SplendorState, Dict[str, int]]:
        """
        Reset the environment - Create a new game.

        :return: the initial state of a new game and the id (turn) of
                 my agent.
        :note: the order of turns in randomly chosen each time reset is called.
        """
        state = self.game_rule.initialGameState()

        random.shuffle(self.agents)
        self.my_turn = random.randint(0, self.number_of_players - 1)

        self.turns_gen = cycle(range(self.number_of_players))

        self._set_opponents_ids()

        self.turn = next(self.turns_gen)
        self.state = state

        _, self.state = self._simulate_opponents()

        return (self.state, {"my_id": self.my_turn})

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Run one time-step of the environment's dynamics.

        :param: which action to take.
        :return: The new state (successor), the reward given, a flag indicating whether or
                 not the game ended, truncated (will be ignored), additional
                 information (will be ignored).

        :note: this method returns 2 redundant variables (truncated & info) only in
               order to comply with gym.Env.step signature.
        """
        if action not in range(len(ALL_ACTIONS)):
            raise ValueError(f"The action {action} isn't a valid action")

        self.turn = next(self.turns_gen)
        previous_score = self.state.agents[self.turn].score
        action_to_take = self._build_action(action)

        # generateSuccessor return a reference to the same
        # state object which is updated in-place.
        next_state = self.game_rule.generateSuccessor(
            self.state, action_to_take, self.turn
        )
        self.state = next_state
        current_score = self.state.agents[self.turn].score

        if current_score >= 15:
            reward = 100
        else:
            reward = current_score - previous_score

        # this refers to the next_state.
        terminated_by_me = self.game_rule.gameEnds()

        # simulate opponents until my next turn
        terminated_by_opponents, next_state = self._simulate_opponents()
        terminated = terminated_by_me or terminated_by_opponents

        # TODO: return feature-vector of next_state (as np.array) instead of SplendorState.
        #       this will also remove a gym warning specifically about that.

        return (next_state, reward, terminated, False, {})

    def render(self):
        # Don't render anything.
        pass

    def _set_opponents_ids(self):
        ids = list(range(self.number_of_players))
        ids.remove(self.my_turn)

        for agent, agent_id in zip(self.agents, ids):
            agent.id = agent_id

    def _simulate_opponents(self) -> Tuple[bool, SplendorState]:
        """
        Simulate the opponents moves from the current turn until self.my_turn

        :return: whether or not the game has ended prior to the turn of self.my_turn
        """
        while self.turn != self.my_turn and not self.game_rule.gameEnds():
            available_actions = self.game_rule.getLegalActions(self.state, self.turn)
            action = self.agents[self.turn].SelectAction(
                available_actions, self.state, self.game_rule
            )
            self.state = self.game_rule.generateSuccessor(self.state, action, self.turn)
            self.turn = next(self.turns_gen)

        return self.game_rule.gameEnds(), self.state

    def _valid_position(self, state: SplendorState, position: CardPosition) -> bool:
        if position.tier not in range(NUMBER_OF_DECKS):
            return False
        if (
            position.card_index not in range(MAX_NUMBER_OF_CARDS_FROM_DECK)
            or state.board.dealt[position.tier][position.card_index] is None
        ):
            return False
        return True

    def _valid_reserved_position(
        self, state: SplendorState, position: CardPosition, agent_index: int
    ) -> bool:
        return position.reserved_index in range(
            len(state.agents[agent_index].cards[RESERVED])
        )

    def _build_action(
        self,
        action_index: int,
        state: Optional[SplendorState] = None,
        agent_index: Optional[int] = None,
    ):
        """
        Construct the action to be taken from it's action index in the ALL_ACTION list.
        :return: the corresponding action to the action_index, in the format required
                 by SplendorGameRule.
        """
        if action_index not in range(len(ALL_ACTIONS)):
            raise ValueError(f"The action {action_index} isn't a valid action")

        if state is None:
            state = self.state

        if agent_index is None:
            agent_index = self.turn

        action = ALL_ACTIONS[action_index]

        potential_nobles = self.game_rule.get_potential_nobles(state, agent_index)
        noble = (
            state.board.nobles[action.noble_index]
            if action.noble_index
            and action.noble_index in range(len(state.board.nobles))
            else None
        )
        card = (
            state.board.dealt[action.position.tier][action.position.card_index]
            if action.position and self._valid_position(state, action.position)
            else None
        )
        reserved_card = (
            state.agents[agent_index].cards[RESERVED][action.position.reserved_index]
            if action.position
            and self._valid_reserved_position(state, action.position, agent_index)
            else None
        )

        match action.type:
            case ActionType.PASS:
                action_to_execute = {
                    "type": "pass",
                    "noble": noble,
                }
            case ActionType.COLLECT_SAME:
                action_to_execute = {
                    "type": "collect_same",
                    "noble": noble,
                    "collected_gems": action.collected_gems,
                    "returned_gems": action.returned_gems,
                }
            case ActionType.COLLECT_DIFF:
                action_to_execute = {
                    "type": "collect_diff",
                    "noble": noble,
                    "collected_gems": action.collected_gems,
                    "returned_gems": action.returned_gems,
                }
            case ActionType.RESERVE:
                action_to_execute = {
                    "type": "reserve",
                    "noble": noble,
                    "card": card,
                    "collected_gems": action.collected_gems,
                    "returned_gems": action.returned_gems,
                }
            case ActionType.BUY_AVAILABLE:
                action_to_execute = {
                    "type": "buy_available",
                    "noble": noble,
                    "card": card,
                    "returned_gems": card.cost,
                }
            case ActionType.BUY_RESERVE:
                action_to_execute = {
                    "type": "buy_reserve",
                    "noble": noble,
                    "card": reserved_card,
                    "returned_gems": reserved_card.cost,
                }

        return action_to_execute
