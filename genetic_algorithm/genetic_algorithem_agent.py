from copy import deepcopy
from pathlib import Path

from Engine.Splendor.features import extract_metrics
from Engine.template import Agent
import numpy as np

from genes import ManagerGene, StrategyGene 



MANAGER_PATH = "manager.npy"
STRATEGY_1_PATH = "strategy1.npy"
STRATEGY_2_PATH = "strategy2.npy"
STRATEGY_3_PATH = "strategy3.npy"



class GeneAlgoAgent(Agent):
    def __init__(self, _id, manager=None,
                 strategy1=None, strategy2=None, strategy3=None):
        super().__init__(_id)

        if manager is None:
            self._manager_gene = ManagerGene.load(MANAGER_PATH)
        else:
            self._manager_gene = manager

        if strategy1 is None:
            self._strategy_gene_1 = StrategyGene.load(STRATEGY_1_PATH)
        else:
            self._strategy_gene_1 = strategy1

        if strategy2 is None:
            self._strategy_gene_2 = StrategyGene.load(STRATEGY_2_PATH)
        else:
            self._strategy_gene_2 = strategy2

        if strategy3 is None:
            self._strategy_gene_3 = StrategyGene.load(STRATEGY_3_PATH)
        else:
            self._strategy_gene_3 = strategy3

        self._strategies = [
            self._strategy_gene_1,
            self._strategy_gene_2,
            self._strategy_gene_3,
        ]


    def __hash__(self):
        return hash(f"{id(self)}:{id(self._strategies)}")


    def save(self, folder: Path):
        self._manager_gene.save(folder / MANAGER_PATH)
        self._strategy_gene_1.save(folder / STRATEGY_1_PATH)
        self._strategy_gene_2.save(folder / STRATEGY_2_PATH)
        self._strategy_gene_3.save(folder / STRATEGY_3_PATH)


    def evaluate_action(self, strategy, action, game_state, game_rule):
        # Fix this - use `game_rule.generate_predecessor` or an optimized metric
        #            extraction method which gets both the action and the state.
        original_agent_state = deepcopy(game_state.agents[self.id])
        original_board = deepcopy(game_state.board)

        next_state = game_rule.generateSuccessor(game_state, action, self.id)
        # game_state.agents[agent_id] = original_agent_state
        # game_state.board = original_board
        next_metrics = extract_metrics(next_state, self.id)
        evaluation = strategy.evaluate_state(next_metrics)
        game_rule.generatePredecessor(game_state, action, self.id)

        return evaluation


    def SelectAction(self, actions, game_state, game_rule):
        if not actions:
            raise Exception("Cannot play, no actions")

        metrics = extract_metrics(game_state, self.id)
        strategy = self._manager_gene.select_strategy(metrics,
                                                      self._strategies)
        best_action = None
        best_action_value = None

        for action in actions:
            action_value = self.evaluate_action(strategy, action, game_state, game_rule)

            if best_action is None or action_value > best_action_value:
                best_action = action
                best_action_value = action_value

        return best_action


myAgent = GeneAlgoAgent
