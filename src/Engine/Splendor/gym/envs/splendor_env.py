"""
Implementation of Splendor as a gym.Env
"""

from typing import Dict, Literal, List, Tuple, Optional, Any

import numpy as np
import gymnasium as gym

from Engine.template import Agent
from Engine.Splendor.splendor_model import SplendorState, SplendorGameRule
from Engine.Splendor.utils import LimitRoundsGameRule
from Engine.Splendor import splendor_utils
from Engine.Splendor import features

from .actions import ALL_ACTIONS, ActionType, Action, CardPosition
from .utils import create_legal_actions_mask, create_action_mapping


class SplendorEnv(gym.Env):
    def __init__(
        self,
        agents: List[Agent],
        shuffle_turns: bool = True,
        fixed_turn: Optional[int] = None,
        render_mode: Optional[Any] = None,
    ):
        """
        Create a new environment, which simulates the game of Splendor by
        using SplendorGameRule.

        :param agents: a list of all the opponents.
        :param shuffle_turns: whether or not to shuffle the order of turns
                              every time reset() is called.
        :param fixed_turn: fix the turn of the player, if isn't supplied or
                           out of range the turn of the player would be
                           randomly chosen during reset().
        :param render_mode: ignored (it's here for compatibility with gym).

        :note: when using this environment one can use gymnasium.spaces.utils.flatdim
               in order to understand the dimensions of the environment.
               for example:
                    from gymnasium.spaces.utils import flatdim
                    num_actions = flatdim(env.action_space)
                    flatten_state_features_length = flatdim(env.observation_space)
               useful later for the DQN in order to define it's input & output dimensions.

               when creating an env using gym.make(...) gymnasium will automatically wrap
               the Env object with some wrappers. in order for you to access the base Env
               object without losing ones sanity you can use the .unwrapped property which
               returns a reference to the base Env without any wrappers. this could be
               useful when you want to use some custom methods of this Env such as
               build_action & get_legal_actions_mask.
        """
        super().__init__()

        self.fixed_turn = fixed_turn
        self.shuffle_turns = shuffle_turns
        self.agents = agents
        self.number_of_players = len(self.agents) + 1
        self.game_rule = LimitRoundsGameRule(self.number_of_players)

        # define the action_space to be composed of len(ALL_ACTIONS)
        # unique actions.
        self.action_space = gym.spaces.Discrete(len(ALL_ACTIONS))

        # define the observation_space to be a vector of continues values of length
        # of features.METRICS_WITH_CARDS_SIZE.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=features.METRICS_WITH_CARDS_SHAPE,
        )

        # this default value will be overridden by reset().
        self.my_turn: int = -1

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[SplendorState, Dict[str, int]]:
        """
        Reset the environment - Create a new game.

        :param seed: the seed to use in np_random.
        :param options: ignored, both this parameter & seed are passed in
                        order to comply with gym.Env signature.
        :return: the initial state of a new game and the id (turn) of
                 my agent.
        :note: the order of turns in randomly chosen each time reset is called.
        """
        self.game_rule = LimitRoundsGameRule(self.number_of_players)

        if self.shuffle_turns:
            np.random.shuffle(self.agents)

        if self.fixed_turn is None or (
            self.fixed_turn not in range(self.number_of_players)
        ):
            self.my_turn = np.random.randint(0, self.number_of_players)
        else:
            self.my_turn = self.fixed_turn

        self._set_opponents_ids()

        self._simulate_opponents()

        return (
            self._vectorize(self.state, self.observation_space.shape, self.my_turn),
            {"my_id": self.my_turn},
        )

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
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

        previous_score = self.state.agents[self.my_turn].score

        legal_actions = self.game_rule.getLegalActions(self.state, self.my_turn)

        mapping = create_action_mapping(legal_actions, self.state, self.my_turn)

        action_to_take = mapping[action]

        legal_actions_mask: np.array = self.get_legal_actions_mask()

        if legal_actions_mask[action] == 0:
            raise ValueError(f"the action {action} ({action_to_take}) is illegal!")

        self.game_rule.update(action_to_take)
        current_score = self.state.agents[self.my_turn].score

        # simulate opponents until my next turn
        terminated, next_state = self._simulate_opponents()

        is_leading = (
            np.array(
                [self.state.agents[i].score for i in range(self.number_of_players)]
            ).argmax()
            == self.my_turn
        )

        if current_score >= 15 and is_leading:
            reward = 100
        else:
            reward = current_score - previous_score

        return (
            self._vectorize(next_state, self.observation_space.shape, self.my_turn),
            reward,
            terminated,
            False,
            {},
        )

    def render(self):
        # Don't render anything.
        pass

    def get_legal_actions_mask(self) -> np.array:
        """
        Create an array of shape (len(ALL_ACTIONS),) whose values are 0's or 1's.
        If the at the i'th index the mask[i] == 1 then the i'th action is legal,
        otherwise it's illegal (The legal actions are based on SplendorGameRule).
        """
        legal_actions = self.game_rule.getLegalActions(self.state, self.my_turn)
        legal_actions_mask: np.array = create_legal_actions_mask(
            legal_actions, self.state, self.my_turn
        )

        return legal_actions_mask

    def _get_opponent_by_turn(self, turn: int) -> Agent:
        """
        :return: the agent whose ID is equal to turn.
        """
        for agent in self.agents:
            if agent.id == turn:
                return agent

        raise ValueError(f"Invalid turn ({turn})")

    @staticmethod
    def _vectorize(state: SplendorState, shape: Tuple[int, ...], turn: int) -> np.array:
        """
        extract the features vector out of the given state.
        """
        return features.extract_metrics_with_cards(state, turn).astype(np.float32)

    def _set_opponents_ids(self):
        """
        assign IDs to all the agents of the opponents according to the turns
        ordering.
        """
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
            agent = self._get_opponent_by_turn(self.turn)
            action = agent.SelectAction(available_actions, self.state, self.game_rule)
            self.game_rule.update(action)

        return self.game_rule.gameEnds(), self.state

    @property
    def turn(self):
        return self.game_rule.current_agent_index

    @property
    def state(self):
        return self.game_rule.current_game_state
