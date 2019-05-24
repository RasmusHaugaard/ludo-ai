import math
import random

import numpy as np
from progressbar import ProgressBar, Percentage

from pyludo import LudoGame


class BaseTournamentSelection:
    name = "base_tournament"
    args = []
    population = np.array([])
    progress_bar = None
    cur_tournament_count = None
    current_generation = 0
    total_game_count = 0

    def __init__(self, Player, population_size, pop_init, recombine, mutate):
        self.Player = Player
        self.population = pop_init(population_size)
        for chromosome in self.get_flat_pop():
            chromosome[:] = Player.normalize(chromosome)
        self.population_size = population_size
        assert (population_size % 4 == 0)
        self.tournaments_per_generation = population_size // 4
        self.mutate = mutate
        self.recombine = recombine

    def get_flat_pop(self):
        return self.population.reshape((-1, self.population.shape[-1]))

    def play_tournament(self, chromosome_ids, game_count):
        flat_pop = self.get_flat_pop()
        chromosomes = flat_pop[chromosome_ids]
        players = [self.Player(chromosome) for chromosome in chromosomes]
        tournament_player_ids = {}
        for i, player in enumerate(players):
            tournament_player_ids[player] = i
        win_rates = np.zeros(4)
        for _ in range(game_count):
            random.shuffle(players)
            game = LudoGame(players)
            winner = players[game.play_full_game()]
            win_rates[tournament_player_ids[winner]] += 1
        ranked_chromosome_ids = chromosome_ids[np.argsort(-win_rates)]
        children = self.recombine(*flat_pop[ranked_chromosome_ids[:2]])
        children = [self.mutate(child) for child in children]
        children = [self.Player.normalize(child) for child in children]
        flat_pop[ranked_chromosome_ids[2:]] = children

        self.total_game_count += game_count
        self.cur_tournament_count += 1
        self.progress_bar.update(self.cur_tournament_count)

    def step(self, generation_count=1):
        total_tournament_count = generation_count * self.tournaments_per_generation
        self.cur_tournament_count = 0
        text = "Generation {}, playing {} tournaments now...".format(self.current_generation, total_tournament_count)
        self.progress_bar = ProgressBar(widgets=[text, Percentage()], maxval=total_tournament_count).start()
        for _ in range(generation_count):
            self.next_generation()
            self.current_generation += 1
        self.progress_bar.finish()

    def next_generation(self):
        pass


class TournamentSelection(BaseTournamentSelection):
    name = "tournament"
    args = [("population_size", int), ("games_per_tournament", int)]

    def __init__(self, Player, pop_init, recombine, mutate, population_size, games_per_tournament):
        super(TournamentSelection, self).__init__(Player, population_size, pop_init, recombine, mutate)
        self.games_per_tournament = games_per_tournament
        self.all_chromosome_ids = np.arange(population_size)

    def next_generation(self):
        np.random.shuffle(self.all_chromosome_ids)
        for tournament_id in range(self.population_size // 4):
            chromosome_ids = self.all_chromosome_ids[tournament_id * 4:tournament_id * 4 + 4]
            self.play_tournament(chromosome_ids, self.games_per_tournament)


class CellularTournamentSelection(BaseTournamentSelection):
    name = "cellular_tournament"
    args = [("population_size", int), ("games_per_tournament", int)]

    def __init__(self, Player, pop_init, recombine, mutate, population_size, games_per_tournament):
        grid_size = int(round(math.sqrt(population_size)))
        assert population_size % grid_size == 0
        assert grid_size % 2 == 0
        super(CellularTournamentSelection, self).__init__(Player, population_size, pop_init, recombine, mutate)
        self.population = self.population.reshape((grid_size, grid_size, -1))
        self.grid_size = grid_size
        self.games_per_tournament = games_per_tournament

    def next_generation(self, generation_count=1):
        off_x, off_y = [(0, 0), (0, 1), (1, 1), (1, 0)][self.current_generation % 4]
        for x in range(0, self.grid_size, 2):
            for y in range(0, self.grid_size, 2):
                chromosome_ids = np.array([
                    ((y + dy + off_y) % self.grid_size) * self.grid_size + (x + dx + off_x) % self.grid_size
                    for dx, dy in ((0, 0), (0, 1), (1, 1), (1, 0))
                ])
                self.play_tournament(chromosome_ids, self.games_per_tournament)


class IslandTournamentSelection(BaseTournamentSelection):
    name = "island_tournament"
    args = [
        ("island_count", int), ("chromosomes_per_island", int), ("generations_per_epoch", int),
        ("migration_count", int), ("games_per_tournament", int)
    ]

    def __init__(self, Player, pop_init, recombine, mutate, island_count, chromosomes_per_island, generations_per_epoch,
                 migration_count, games_per_tournament):
        assert (chromosomes_per_island % 4 == 0)
        super(IslandTournamentSelection, self).__init__(Player, island_count * chromosomes_per_island, pop_init,
                                                        recombine, mutate)
        self.island_count = island_count
        self.chromosome_per_island = chromosomes_per_island
        self.population = self.population.resize((island_count, chromosomes_per_island, -1))
        self.migration_count = migration_count
        self.generations_per_epoch = generations_per_epoch
        self.games_per_tournament = games_per_tournament
        self.all_island_chromosome_ids = np.arange(chromosomes_per_island)

    def next_generation(self):
        for island_id in range(self.island_count):
            np.random.shuffle(self.all_island_chromosome_ids)
            for tournament_id in range(self.chromosome_per_island // 4):
                chromosome_ids = self.all_island_chromosome_ids[tournament_id * 4:tournament_id * 4 + 4]
                chromosome_ids += island_id * self.chromosome_per_island
                self.play_tournament(chromosome_ids, self.games_per_tournament)
        if self.current_generation % self.generations_per_epoch == 0:
            self.migrate()

    def migrate(self):
        # pick self.migration_count on each island and shuffle those chromosomes
        migrant_ids = np.array((self.island_count, self.migration_count), dtype=np.int)
        for island_id in range(self.island_count):
            migrant_ids[island_id] = np.random.choice(self.all_island_chromosome_ids, self.migration_count,
                                                      replace=False)
        old_migrant_ids = migrant_ids.reshape(-1)
        new_migrant_ids = old_migrant_ids.copy()
        np.random.shuffle(new_migrant_ids)
        self.get_flat_pop()[new_migrant_ids] = self.get_flat_pop()[old_migrant_ids]


def get_selection(name):
    selections = [TournamentSelection, CellularTournamentSelection, IslandTournamentSelection]
    selector_map = {}
    for selection in selections:
        selector_map[selection.name] = selection
    return selector_map[name]
