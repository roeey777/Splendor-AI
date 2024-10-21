"""
Genetic Algorithm based agent.
"""

from pathlib import Path
from typing import override

import numpy as np

from splendor.agents.our_agents.genetic_algorithm.genes import ManagerGene, StrategyGene
from splendor.splendor.features import extract_metrics, normalize_metrics
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class GeneAlgoAgent(Agent):
    """
    Agent which plays according to a "plan" reached by genetic algorithm.
    This agent is also used to represent an individual in the evolution
    process.
    We decided to describe an agents plan using 4 genes: 3 of them are used to
    evaluate actions and choose one from an available actions list and the
    forth used to assess the current situation and choose one of the 3
    strategies.
    """

    MANAGER_PATH = Path(__file__).parent / "manager.npy"
    STRATEGY_1_PATH = Path(__file__).parent / "strategy1.npy"
    STRATEGY_2_PATH = Path(__file__).parent / "strategy2.npy"
    STRATEGY_3_PATH = Path(__file__).parent / "strategy3.npy"
    INVALID_POPULATION_ID = -1

    def __init__(
        self,
        _id: int,
        manager: ManagerGene | None = None,
        strategy1: StrategyGene | None = None,
        strategy2: StrategyGene | None = None,
        strategy3: StrategyGene | None = None,
    ) -> None:
        super().__init__(_id)

        if manager is None:
            self.manager_gene = ManagerGene.load(self.MANAGER_PATH)
        else:
            self.manager_gene = manager

        if strategy1 is None:
            self.stategy_gene_1 = StrategyGene.load(self.STRATEGY_1_PATH)
        else:
            self.stategy_gene_1 = strategy1

        if strategy2 is None:
            self.stategy_gene_2 = StrategyGene.load(self.STRATEGY_2_PATH)
        else:
            self.stategy_gene_2 = strategy2

        if strategy3 is None:
            self.stategy_gene_3 = StrategyGene.load(self.STRATEGY_3_PATH)
        else:
            self.stategy_gene_3 = strategy3

        self._strategies = (
            self.stategy_gene_1,
            self.stategy_gene_2,
            self.stategy_gene_3,
        )

        self.population_id = self.INVALID_POPULATION_ID

    def save(self, folder: Path) -> None:
        """
        Saves the genes of the given agent to the provided folder.
        Used for evolution.
        """
        self.manager_gene.save(folder / self.MANAGER_PATH.name)
        self.stategy_gene_1.save(folder / self.STRATEGY_1_PATH.name)
        self.stategy_gene_2.save(folder / self.STRATEGY_2_PATH.name)
        self.stategy_gene_3.save(folder / self.STRATEGY_3_PATH.name)

    def evaluate_action(
        self,
        strategy: StrategyGene,
        action: ActionType,
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> float:
        """
        Evaluates an `action` by the metrcis of the game's state after the
        action. The `strategy` is used to evaluate the state.
        """
        next_state = game_rule.generateSuccessor(game_state, action, self.id)
        next_metrics = normalize_metrics(extract_metrics(next_state, self.id))
        evaluation = strategy.evaluate_state(next_metrics)
        game_rule.generatePredecessor(game_state, action, self.id)

        return evaluation

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """
        Method used by the game's engine when running a game with this agent.
        """
        if not actions:
            raise RuntimeError("Cannot play, no actions")

        metrics = normalize_metrics(extract_metrics(game_state, self.id))
        strategy = self.manager_gene.select_strategy(metrics, self._strategies)
        best_action = None
        best_action_value = -np.inf

        for action in actions:
            action_value = self.evaluate_action(strategy, action, game_state, game_rule)

            if action_value > best_action_value:
                best_action = action
                best_action_value = action_value

        assert best_action is not None
        return best_action


# Required for the game engine to use this agent
myAgent = GeneAlgoAgent  # pylint: disable=invalid-name
