"""
Genetic algorithm based agent evolution program.
"""

import shutil
from csv import writer as csv_writer
from datetime import datetime
from itertools import starmap
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.genetic_algorithm.genes import (
    Gene,
    ManagerGene,
    StrategyGene,
)
from splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent import (
    GeneAlgoAgent,
)
from splendor.game import Game
from splendor.splendor import features
from splendor.splendor.utils import LimitRoundsGameRule

from .argument_parsing import parse_args
from .constants import (
    CHILDREN_PER_MATING,
    DEPENDECY_DEGREE,
    FOLDER_FORMAT,
    FOUR_PLAYERS,
    GENERATIONS,
    MUTATION_RATE,
    PARENTS_PER_MATING,
    PLAYERS_OPTIONS,
    POPULATION_SIZE,
    STATS_FILE,
    STATS_HEADERS,
    WINNER_BONUS,
    WORKING_DIR,
)

GamesStats = list[list[int | float | str]]

MAX_PROCESS = cpu_count() // 2


def mutate(gene: Gene, progress: float, mutate_rate: float) -> None:
    """
    Mutates a single gene.
    """

    def _mutate(value: float) -> float:
        """
        Mutation method is based on the following article (page 112)
        http://web.ist.utl.pt/adriano.simoes/tese/referencias/Michalewicz%20Z.%20Genetic%20Algorithms%20+%20Data%20Structures%20=%20Evolution%20Programs%20%283ed%29.PDF
        """
        diff = value - np.random.choice((gene.LOWER_BOUND, gene.UPPER_BOUND))
        power = (1 - progress) ** DEPENDECY_DEGREE
        return value - diff * (1 - np.random.rand() ** power)

    gene.mutate(mutate_rate, _mutate)


def mutate_population(
    population: list[GeneAlgoAgent], progress: float, mutation_rate: float
) -> None:
    """
    Mutates the genes of the population.
    """
    for agent in population:
        mutate(agent.manager_gene, progress, mutation_rate)
        mutate(agent.stategy_gene_1, progress, mutation_rate)
        mutate(agent.stategy_gene_2, progress, mutation_rate)
        mutate(agent.stategy_gene_3, progress, mutation_rate)


def _crossover(dna1: NDArray, dna2: NDArray) -> tuple[NDArray, NDArray]:
    """
    Crossover method is based on the following article (page 9)
    https://www.cs.us.es/~fsancho/ficheros/IA2019/TheContinuousGeneticAlgorithm.pdf
    """
    split_point = np.random.randint(len(dna1))
    mix_coefficient = np.random.rand()
    diff = dna1[split_point] - dna2[split_point]
    new_value_1 = dna1[split_point] - mix_coefficient * diff
    new_value_2 = dna2[split_point] + mix_coefficient * diff
    child1 = np.hstack((dna1[:split_point], [new_value_1], dna2[split_point + 1 :]))
    child2 = np.hstack((dna2[:split_point], [new_value_2], dna1[split_point + 1 :]))
    return child1, child2


def crossover(mom: Gene, dad: Gene) -> tuple[Gene, Gene]:
    """
    Executes crossover between 2 genes, which produces 2 children.
    """
    cls = mom.__class__
    if not isinstance(dad, cls):
        raise TypeError("Crossover works only between genes of the same type")

    # this assertion is only for mypy.
    assert cls.SHAPE is not None

    child_dna_1: NDArray | tuple[NDArray, ...]
    child_dna_2: NDArray | tuple[NDArray, ...]

    match len(cls.SHAPE):
        case 1:
            child_dna_1, child_dna_2 = _crossover(mom.raw_dna, dad.raw_dna)
            return cls(child_dna_1), cls(child_dna_2)
        case 2:
            children_dna = starmap(
                _crossover, zip(mom.raw_dna.T, dad.raw_dna.T, strict=True)
            )
            child_dna_1, child_dna_2 = zip(*children_dna, strict=True)
            return (
                cls(np.vstack(cast(tuple[NDArray, ...], child_dna_1)).T),
                cls(np.vstack(cast(tuple[NDArray, ...], child_dna_2)).T),
            )

    raise ValueError(f"Unsupported DNA shape for crossover {cls.SHAPE}")


def mate(parents: list[GeneAlgoAgent], population_size: int) -> list[GeneAlgoAgent]:
    """
    Creates new individual by randomly choosing 2 parents and mating them till
    we got enough individuals.
    """
    parents_array = np.array(parents)
    children = []
    matings = (population_size - len(parents)) // CHILDREN_PER_MATING
    for _ in range(matings):
        mom, dad = np.random.choice(parents_array, PARENTS_PER_MATING, False)
        managers = cast(
            list[ManagerGene], crossover(mom.manager_gene, dad.manager_gene)
        )
        strategies_1 = cast(
            list[StrategyGene], crossover(mom.stategy_gene_1, dad.stategy_gene_1)
        )
        strategies_2 = cast(
            list[StrategyGene], crossover(mom.stategy_gene_2, dad.stategy_gene_2)
        )
        strategies_3 = cast(
            list[StrategyGene], crossover(mom.stategy_gene_3, dad.stategy_gene_3)
        )
        for i in range(CHILDREN_PER_MATING):
            children.append(
                GeneAlgoAgent(
                    0, managers[i], strategies_1[i], strategies_2[i], strategies_3[i]
                )
            )

    return children


def single_game(agents: list[GeneAlgoAgent]) -> tuple[Game, dict]:
    """
    Runs a single game of Splendor (with the Engine) using the given agents.
    """
    agents_array = np.array(agents)
    np.random.shuffle(agents_array)
    agents = agents_array.tolist()
    names = []
    for i, agent in enumerate(agents):
        agent.id = i
        names.append(str(i))

    game = Game(
        LimitRoundsGameRule,
        agents,
        len(agents),
        seed=np.random.randint(int(1e8), dtype=int),
        agents_namelist=names,
    )
    return game, game.Run()


def _evaluate_multiprocess(
    population: list[GeneAlgoAgent],
    players_count: int,
) -> list[tuple[Game, dict]]:
    games = len(population) // players_count
    if players_count == FOUR_PLAYERS:
        agents_generator = (
            population[i : i + FOUR_PLAYERS]
            for i in range(0, len(population), FOUR_PLAYERS)
        )
    else:
        agents_generator = (population[i::games] for i in range(games))

    with Pool(MAX_PROCESS) as pool:
        return pool.map(single_game, agents_generator)


def _evaluate(
    population: list[GeneAlgoAgent],
    players_count: int,
    quiet: bool,
) -> list[tuple[Game, dict]]:
    results: list[tuple[Game, dict]] = []
    games = len(population) // players_count

    for i in range(games):
        if not quiet:
            print(
                f"        game number {i+1} "
                f"({datetime.now().strftime(FOLDER_FORMAT)})"
            )

        if players_count == FOUR_PLAYERS:
            agents = population[i * FOUR_PLAYERS : (i + 1) * FOUR_PLAYERS]
        else:
            agents = population[i::games]
        results.append(single_game(agents))

    return results


def evaluate(
    population: list[GeneAlgoAgent],
    quiet: bool,
    multiprocess: bool,
) -> tuple[list[float], GamesStats]:
    """
    Measures the fitness of each individual by having them play against each
    other. Each individual plays in 3 games with 1,2 and 3 rivals.
    """
    games_stats: GamesStats = []
    evaluation: list[float] = [0] * len(population)

    for players_count in PLAYERS_OPTIONS:
        if not quiet:
            print(f"    evaluating games of {players_count} players")

        if multiprocess:
            results = _evaluate_multiprocess(population, players_count)
        else:
            results = _evaluate(population, players_count, quiet)

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
                np.mean(tuple(result["scores"].values())),
            ]
            stats.extend(result["scores"].get(i, "None") for i in range(FOUR_PLAYERS))
            cards_in_play = zip(
                game.game_rule.current_game_state.board.decks,
                game.game_rule.current_game_state.board.dealt,
                strict=True,
            )
            stats.extend(
                len(deck) + len(tuple(filter(None, dealt)))
                for deck, dealt in cards_in_play
            )
            games_stats.append(stats)

    return evaluation, games_stats


def sort_by_fitness(
    population: list[GeneAlgoAgent],
    folder: Path,
    message: str,
    quiet: bool,
    multiprocess: bool,
) -> GamesStats:
    """
    Sort the individuals of the population based on their fitness score.

    :param population: list of all the individuals comprizing the entire population.
    :param folder: where to store the fittest individual of the population.
    :param message: a message to print.
    :param quiet: should print the given message or stay silent.
    :param multiprocess: should the games simulations uses multi-processing or a single-process.
    :return: the games statistics.
    """
    if not quiet:
        print(message)

    for i, agent in enumerate(population):
        agent.population_id = i

    evaluation, games_stats = evaluate(population, quiet, multiprocess)
    population.sort(key=lambda agent: evaluation[agent.population_id], reverse=True)

    if not quiet:
        print(
            "    Saving the best agent " f"({evaluation[population[0].population_id]})"
        )
    folder.mkdir()
    population[0].save(folder)

    return games_stats


def generate_initial_population(population_size: int) -> list[GeneAlgoAgent]:
    """
    Creates agents with random genes.
    """
    return [
        GeneAlgoAgent(
            0,
            ManagerGene.random(),
            StrategyGene.random(),
            StrategyGene.random(),
            StrategyGene.random(),
        )
        for _ in range(population_size)
    ]


def evolve(  # noqa: PLR0913,PLR0917
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    working_dir: Path = WORKING_DIR,
    seed: int | None = None,
    quiet: bool = False,
    multiprocess: bool = False,
) -> list[GeneAlgoAgent]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """
    Genetic algorithm evolution process.
    In each generation `selection_size` are kept and used for mating.
    Returns the top `return_size` individuals of the last generation.
    """
    start_time = datetime.now()
    selection_size = (population_size // 3) or 2
    return_size = (population_size // 12) or 1
    if seed is not None:
        np.random.seed(seed)

    folder = working_dir / start_time.strftime(FOLDER_FORMAT)
    folder.mkdir()
    shutil.copy(features.__file__, folder)

    if not quiet:
        print(f"({folder.name}) Starting evolution with")
        print(f"    population: {population_size}")
        print(f"    selection:  {selection_size}")

    population = generate_initial_population(population_size)

    with Path.open(
        folder / STATS_FILE, "w", newline="\n", encoding="ascii"
    ) as stats_file:
        stats_csv = csv_writer(stats_file)
        stats_csv.writerow(STATS_HEADERS)

        for generation in range(1, generations + 1):
            progress = generation / generations
            games_stats = sort_by_fitness(
                population,
                folder / str(generation),
                f"Gen {generation}",
                quiet,
                multiprocess,
            )

            for stats in games_stats:
                stats.insert(0, generation)
                stats_csv.writerow(stats)

            parents = population[:selection_size]
            parents_array = np.array(parents)
            np.random.shuffle(parents_array)
            children = mate(parents_array.tolist(), population_size)
            mutate_population(children, progress, mutation_rate)
            population = parents + children
            population_array = np.array(population)
            np.random.shuffle(population_array)
            population = population_array.tolist()

        games_stats = sort_by_fitness(
            population, folder / "final", "Final", quiet, multiprocess
        )
        for stats in games_stats:
            stats.insert(0, "final")
            stats_csv.writerow(stats)

    if not quiet:
        print(f"Done (run time was {datetime.now() - start_time})")
    return population[:return_size]


def main() -> None:
    """
    entry-point for the ``evolve`` console script.
    """
    options = parse_args()
    evolve(**options)


if __name__ == "__main__":
    main()
