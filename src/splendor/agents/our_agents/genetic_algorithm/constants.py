"""
Genetic algorithm evolution constants.
"""

from pathlib import Path

POPULATION_SIZE = 24  # 60
GENERATIONS = 100
MUTATION_RATE = 0.2
DEPENDECY_DEGREE = 3
FOUR_PLAYERS = 4
PLAYERS_OPTIONS = (2, 3, 4)
CHILDREN_PER_MATING = 2
PARENTS_PER_MATING = 2
WORKING_DIR = Path().absolute()
FOLDER_FORMAT = "%y-%m-%d_%H-%M-%S"
WINNER_BONUS = 0
STATS_FILE = "stats.csv"
STATS_HEADERS = (
    "generation",
    "players_count",
    "rounds_count",
    "nobles_taken",
    "scores_mean",
    "player1_score",
    "player2_score",
    "player3_score",
    "player4_score",
    "tier1_left",
    "tier2_left",
    "tier3_left",
)
