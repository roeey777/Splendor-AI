from datetime import datetime
from pathlib import Path

from Engine.game import Game
from Engine.Splendor.splendor_model import SplendorGameRule
import numpy as np

from genes import Gene, ManagerGene, StrategyGene
from genetic_algorithem_agent import GeneAlgoAgent


GENERATIONS = 100
MUTATE_RATE = 0.2
DEPENDECY_DEGREE = 3
ROUNDS_LIMIT = 100
FOUR_PLAYERS = 4
PLAYERS_OPTIONS = (2, 3, 4)
FOLDER_FORMAT = '%y-%m-%d_%H-%M-%S'
POPULATION_SIZE = 24 # 60
SELECTION_SIZE = (POPULATION_SIZE // 3) or 2
RETURN_SIZE = (POPULATION_SIZE // 12) or 1



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


def mutate(gene: Gene, generation: int, mutate_rate: float = MUTATE_RATE):
    """
    Matates a single gene.
    """
    def _mutate(value):
        """
        Mutation method is based on the following article (page 112)
        http://web.ist.utl.pt/adriano.simoes/tese/referencias/Michalewicz%20Z.%20Genetic%20Algorithms%20+%20Data%20Structures%20=%20Evolution%20Programs%20%283ed%29.PDF
        """
        diff = value - np.random.choice((Gene.LOWER_BOUND, Gene.UPPER_BOUND))
        power = (1 - generation / GENERATIONS) ** DEPENDECY_DEGREE
        return value - diff * (1 - np.random.rand() ** power)

    gene.mutate(mutate_rate, _mutate)


def mutate_population(population: list[GeneAlgoAgent], generation: int):
    """
    Mutates the genes of the population.
    """
    for agent in population:
        mutate(agent._manager_gene, generation)
        mutate(agent._strategy_gene_1, generation)
        mutate(agent._strategy_gene_2, generation)
        mutate(agent._strategy_gene_3, generation)


def single_game(agents):
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
    return game.Run()


def generate_initial_population(population_size: int):
    """
    Creates agents with random genes.
    """
    return [GeneAlgoAgent(0, ManagerGene.random(), StrategyGene.random(),
                          StrategyGene.random(), StrategyGene.random())
            for _ in range(population_size)]


def evaluate(population: GeneAlgoAgent) -> dict[GeneAlgoAgent, int]:
    """
    Measures the fitness of each individual by having them play against each
    other. Each individual plays in 3 games with 1,2 and 3 rivals.
    """
    evaluation = dict.fromkeys(population, 0)
    for players_count in PLAYERS_OPTIONS:
        games = len(population) // players_count
        print(f"    evaluating games of {players_count} players")
        for i in range(games):
            print(f"        game number {i+1} "
                  f"({datetime.now().strftime(FOLDER_FORMAT)})")
            if players_count == FOUR_PLAYERS:
                agents = population[i * FOUR_PLAYERS: (i + 1) * FOUR_PLAYERS]
            else:
                agents = population[i::games]
            result = single_game(agents)
            max_score = max(result["scores"].values())
            for agent in agents:
                evaluation[agent] += result["scores"][agent.id]
                if result["scores"][agent.id] == max_score:
                    evaluation[agent] += 10

    return evaluation


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
            children.append(GeneAlgoAgent(0, managers[i], strategies_1[i],
                                          strategies_2[i], strategies_3[i]))

    return children


def evolve(population_size: int = POPULATION_SIZE,
           selection_size: int = SELECTION_SIZE,
           return_size: int = RETURN_SIZE):
    """
    Gentic algorithm evolution process.
    In each generation `selection_size` are kept and used for mating.
    Returns the top `return_size` individuls of the last generation.
    """
    folder = Path(__file__).parent / datetime.now().strftime(FOLDER_FORMAT)
    folder.mkdir()
    print(f"({folder.name}) Starting evolution with")
    print(f"    population: {population_size}")
    print(f"    selection:  {selection_size}")
    print(f"    (return: {return_size})")
    population = generate_initial_population(population_size)

    for generation in range(GENERATIONS):
        print(f'Gen {generation + 1}')
        evaluation = evaluate(population)
        population.sort(key=lambda agent: evaluation[agent], reverse=True)
        print(f'    Saving the best agent ({evaluation[population[0]]})')
        gen_folder = folder / str(generation)
        gen_folder.mkdir()
        population[0].save(gen_folder)

        if generation + 1 < GENERATIONS:
            parents = population[:selection_size]

            np.random.shuffle(parents)
            children = mate(parents, population_size)

            population = parents + children
            np.random.shuffle(population)
            mutate_population(population, generation)

    return population[:return_size]
