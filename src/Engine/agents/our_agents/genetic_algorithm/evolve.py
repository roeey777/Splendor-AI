from argparse import ArgumentParser
from csv import writer as csv_writer
from datetime import datetime
from multiprocessing import cpu_count, Pool
from pathlib import Path

from Engine.agents.our_agents.genetic_algorithm.genes import Gene, ManagerGene, StrategyGene
from Engine.agents.our_agents.genetic_algorithm.genetic_algorithem_agent import GeneAlgoAgent
from Engine.game import Game
from Engine.Splendor.splendor_model import SplendorGameRule
import numpy as np



POPULATION_SIZE = 24 # 60
GENERATIONS = 100
MUTATION_RATE = 0.2
ROUNDS_LIMIT = 100
DEPENDECY_DEGREE = 3
FOUR_PLAYERS = 4
PLAYERS_OPTIONS = (2, 3, 4)
WORKING_DIR = Path().absolute()
FOLDER_FORMAT = '%y-%m-%d_%H-%M-%S'
# SELECTION = (POPULATION_SIZE // 3) or 2
# RETURN_SIZE = (POPULATION_SIZE // 12) or 1
WINNER_BONUS = 10
MAX_PROCESS = cpu_count() // 2
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


class MyGameRule(SplendorGameRule):
    """
    Wraps `SplendorGameRule`.
    """
    def gameEnds(self):
        """
        Limits the game to `ROUNDS_LIMIT` rounds, so random initial agents
        won't get stuck by accident.
        """
        if all(len(agent.agent_trace.action_reward) == ROUNDS_LIMIT
               for agent in self.current_game_state.agents):
            return True

        return super().gameEnds()



def _crossover(dna1: np.array, dna2: np.array) -> tuple[np.array, np.array]:
    """
    Crossover method is based on the following article (page 9)
    https://www.cs.us.es/~fsancho/ficheros/IA2019/TheContinuousGeneticAlgorithm.pdf
    """
    split_point = np.random.randint(len(dna1))
    mix_coefficient = np.random.rand()
    diff = dna1[split_point] - dna2[split_point]
    new_value_1 = dna1[split_point] - mix_coefficient * diff
    new_value_2 = dna2[split_point] + mix_coefficient * diff
    child1 = np.hstack((dna1[:split_point], [new_value_1], dna2[split_point + 1:]))
    child2 = np.hstack((dna2[:split_point], [new_value_2], dna1[split_point + 1:]))
    return child1, child2


def crossover(mom: Gene, dad: Gene) -> tuple[Gene, Gene]:
    """
    Executes crossover between 2 genes, which produces 2 children.
    """
    cls = mom.__class__
    if not isinstance(dad, cls):
        raise TypeError("Crossover works only between genes of the same type")

    if len(cls.SHAPE) == 1:
        child_dna_1, child_dna_2 = _crossover(mom._dna, dad._dna)
        return cls(child_dna_1), cls(child_dna_2)

    elif len(cls.SHAPE) == 2:
        children_dna = (_crossover(dna1, dna2)
                        for dna1, dna2 in zip(mom._dna.T, dad._dna.T))
        child_dna_1, child_dna_2 = zip(*children_dna)
        return cls(np.vstack(child_dna_1).T), cls(np.vstack(child_dna_2).T)

    raise ValueError(f"Unsupported DNA shape for crossover {cls.SHAPE}")


def mutate(gene: Gene, progress: float, mutate_rate: float):
    """
    Mutates a single gene.
    """
    def _mutate(value):
        """
        Mutation method is based on the following article (page 112)
        http://web.ist.utl.pt/adriano.simoes/tese/referencias/Michalewicz%20Z.%20Genetic%20Algorithms%20+%20Data%20Structures%20=%20Evolution%20Programs%20%283ed%29.PDF
        """
        diff = value - np.random.choice((gene.LOWER_BOUND, gene.UPPER_BOUND))
        power = (1 - progress) ** DEPENDECY_DEGREE
        return value - diff * (1 - np.random.rand() ** power)

    gene.mutate(mutate_rate, _mutate)


def mutate_population(population: list[GeneAlgoAgent],
                      progress: float, mutation_rate: float):
    """
    Mutates the genes of the population.
    """
    for agent in population:
        mutate(agent._manager_gene, progress, mutation_rate)
        mutate(agent._strategy_gene_1, progress, mutation_rate)
        mutate(agent._strategy_gene_2, progress, mutation_rate)
        mutate(agent._strategy_gene_3, progress, mutation_rate)


def single_game(agents) -> tuple[Game, dict]:
    """
    Runs a single game of Splendor (with the Engine) using the given agents.
    """
    names = list()
    for i, agent in enumerate(agents):
        agent.id = i
        names.append(str(i))

    game = Game(MyGameRule, agents, len(agents),
                seed=np.random.randint(1e8, dtype=int),
                agents_namelist=names)
    return game, game.Run()


def generate_initial_population(population_size: int):
    """
    Creates agents with random genes.
    """
    return [GeneAlgoAgent(-1, ManagerGene.random(), StrategyGene.random(),
                          StrategyGene.random(), StrategyGene.random())
            for _ in range(population_size)]


def _evalute_multiprocess(
    population: list[GeneAlgoAgent], players_count: int,
) -> list[tuple[Game, dict]]:
    """
    """
    games = len(population) // players_count
    if players_count == FOUR_PLAYERS:
        agents_generator = (population[i : i + FOUR_PLAYERS]
                            for i in range(0, len(population), FOUR_PLAYERS))
    else:
        agents_generator = (population[i::games]
                            for i in range(games))

    with Pool(MAX_PROCESS) as pool:
        return pool.map(single_game, agents_generator)


def _evalute(
    population: list[GeneAlgoAgent], players_count: int, quiet: bool,
) -> list[tuple[Game, dict]]:
    """
    """
    results = list()
    games = len(population) // players_count

    for i in range(games):
        if not quiet:
            print(f"        game number {i+1} "
                  f"({datetime.now().strftime(FOLDER_FORMAT)})")

        if players_count == FOUR_PLAYERS:
            agents = population[i * FOUR_PLAYERS: (i + 1) * FOUR_PLAYERS]
        else:
            agents = population[i::games]
        results.append(single_game(agents))

    return results


def evaluate(
    population: list[GeneAlgoAgent], quiet: bool, multiprocess: bool,
) -> dict[GeneAlgoAgent, int]:
    """
    Measures the fitness of each individual by having them play against each
    other. Each individual plays in 3 games with 1,2 and 3 rivals.
    """
    pool = list()
    games_stats = list()
    evaluation = [0] * len(population)

    for players_count in PLAYERS_OPTIONS:
        if not quiet:
            print(f"    evaluating games of {players_count} players")

        if multiprocess:
            results = _evalute_multiprocess(population, players_count)
        else:
            results = _evalute(population, players_count, quiet)

        for game, result in results:
            max_score = max(result["scores"].values())
            for agent in game.agents:
                evaluation[agent.population_id] += result["scores"][agent.id]
                if result["scores"][agent.id] == max_score:
                    evaluation[agent.population_id] += WINNER_BONUS

            stats = [
                players_count,
                len(result["actions"]) // players_count,
                players_count + 1 - len(game.game_rule.current_game_state.board.nobles),
                np.mean(tuple(result["scores"].values()))
            ]
            stats.extend(result["scores"].get(i, "None")
                         for i in range(FOUR_PLAYERS))
            cards_in_play = zip(
                game.game_rule.current_game_state.board.decks,
                game.game_rule.current_game_state.board.dealt
            )
            stats.extend(len(deck) + len(tuple(filter(None, dealt)))
                         for deck, dealt in cards_in_play)
            games_stats.append(stats)

    return evaluation, games_stats


def mate(parents: list[GeneAlgoAgent], population_size: int) -> list[GeneAlgoAgent]:
    """
    Creates new individual by randomly choosing 2 parents and mating them till
    we got enough individuals.
    """
    CHILDREN_PER_MATING = 2
    PARENTS_PER_MATING = 2

    children = list()
    matings = (population_size - len(parents)) // CHILDREN_PER_MATING
    for _ in range(matings):
        mom, dad = np.random.choice(parents, PARENTS_PER_MATING, False)
        managers = crossover(mom._manager_gene, dad._manager_gene)
        strategies_1 = crossover(mom._strategy_gene_1, dad._strategy_gene_1)
        strategies_2 = crossover(mom._strategy_gene_2, dad._strategy_gene_2)
        strategies_3 = crossover(mom._strategy_gene_3, dad._strategy_gene_3)
        for i in range(CHILDREN_PER_MATING):
            children.append(GeneAlgoAgent(-1, managers[i], strategies_1[i],
                                          strategies_2[i], strategies_3[i]))

    return children


def sort_by_fitness(
    population: list[GeneAlgoAgent],
    folder: Path,
    message: str,
    quiet: bool,
    multiprocess: bool,
) -> list[list]:
    if not quiet:
        print(message)

    for i, agent in enumerate(population):
        agent.population_id = i

    evaluation, games_stats = evaluate(population, quiet, multiprocess)
    population.sort(key=lambda agent: evaluation[agent.population_id],
                    reverse=True)

    if not quiet:
        print('    Saving the best agent '
              f'({evaluation[population[0].population_id]})')
    folder.mkdir()
    population[0].save(folder)

    return games_stats


def evolve(
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    working_dir: Path = WORKING_DIR,
    seed: int = None,
    quiet: bool = False,
    multiprocess: bool = False,
):
    """
    Genetic algorithm evolution process.
    In each generation `selection_size` are kept and used for mating.
    Returns the top `return_size` individuals of the last generation.
    """
    start_time = datetime.now()
    if seed is not None:
        np.random.seed(seed)

    selection_size = (population_size // 3) or 2
    return_size = (population_size // 12) or 1

    folder = working_dir / start_time.strftime(FOLDER_FORMAT)
    folder.mkdir()
    if not quiet:
        print(f"({folder.name}) Starting evolution with")
        print(f"    population: {population_size}")
        print(f"    selection:  {selection_size}")
    population = generate_initial_population(population_size)

    with open(folder / STATS_FILE, "w", newline="\n") as stats_file:
        stats_csv = csv_writer(stats_file)
        stats_csv.writerow(STATS_HEADERS)

        for generation in range(generations):
            progress = generation / generations
            generation += 1
            games_stats = sort_by_fitness(population,
                                          folder / str(generation),
                                          f'Gen {generation}',
                                          quiet,
                                          multiprocess)

            for stats in games_stats:
                stats.insert(0, generation)
                stats_csv.writerow(stats)

            parents = population[:selection_size]
            np.random.shuffle(parents)
            children = mate(parents, population_size)
            population = parents + children
            np.random.shuffle(population)
            mutate_population(population, progress, mutation_rate)

        games_stats = sort_by_fitness(population,
                                      folder / "final",
                                      "Final",
                                      quiet,
                                      multiprocess)
        for stats in games_stats:
            stats.insert(0, "final")
            stats_csv.writerow(stats)

    if not quiet:
        print(f"Done (run time was {datetime.now() - start_time})")
    return population[:return_size]


def main():
    parser = ArgumentParser(
        prog="evolve",
        description="Evolves a Splendor agent using genetic algorithm.",
    )
    parser.add_argument(
        "-p", "--population-size", default=POPULATION_SIZE, type=int,
        help="Size of the population (should be multiple of 12)",
    )
    parser.add_argument(
        "-g", "--generations", default=GENERATIONS, type=int,
        help="Amount of generations",
    )
    parser.add_argument(
        "-m", "--mutation-rate", default=MUTATION_RATE, type=float,
        help="Probability to mutate (should be in the range [0,1])",
    )
    parser.add_argument(
        "-w", "--working-dir", default=WORKING_DIR, type=Path,
        help="Path to directory to work in (will create a directory with "
             "current timestamp for each run)",
    )
    parser.add_argument(
        "-s", "--seed", type=int,
        help="Seed to set for numpy's random number generator",
    )
    parser.add_argument(
        "--multiprocess", action='store_true',
        help="Use multiprocessing to evolve faster",
    )
    parser.add_argument('-q', '--quiet', action='store_true')

    options = parser.parse_args()
    if options.population_size <= 0 or (options.population_size % 12):
        raise ValueError("To work properly, population size should be a "
                         f"multiple of 12 (not {options.population_size})")
    if options.generations <= 0:
        raise ValueError(f"Invalid amount of generations {options.generations}")
    if not 0 <= options.mutation_rate <= 1:
        raise ValueError(f"Invalid mutation rate value {options.mutation_rate}")

    evolve(**options.__dict__)


if __name__ == '__main__':
    main()
