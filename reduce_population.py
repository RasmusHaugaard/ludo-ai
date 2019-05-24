import os
import argparse
import random

import numpy as np

from pyludo import LudoGame, LudoPlayerRandom
from GAPlayers import get_ga_player


def tournament(chromosomes, Player, game_count):
    players = [Player(chromosome) for chromosome in chromosomes]
    while len(players) < 4:
        players.append(LudoPlayerRandom())
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i
    win_rates = np.zeros(4)
    for _ in range(game_count):
        random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1
    ranked_player_ids = np.argsort(-win_rates)
    for id in ranked_player_ids:
        if id < len(chromosomes):
            return id


def get_required_tournament_count(pop_size, played_tournaments=0):
    if pop_size == 1:
        return played_tournaments
    n = pop_size // 4
    new_pop_size = n + pop_size % 4
    if n == 0:
        n = 1
        new_pop_size = 1
    played_tournaments += n
    return get_required_tournament_count(new_pop_size, played_tournaments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_path", required=True)
    parser.add_argument("--games_per_tournament", type=int, required=True)
    args = parser.parse_args()

    folder_path = os.path.dirname(args.population_path)
    folder_name = os.path.basename(folder_path)
    gen_id = int(os.path.basename(args.population_path).split(".")[0])

    player_name = folder_name.split("+")[0]
    Player = get_ga_player(player_name)

    population = list(np.load(args.population_path))
    N = len(population)
    required_tournament_count = get_required_tournament_count(N)
    required_game_count = required_tournament_count * args.games_per_tournament
    print("required game count:", required_game_count)

    tournaments_played = 0
    while N > 1:
        tournament_count = N // 4
        if tournament_count == 0:
            tournament_count = 1
        print("Currently {} players in the population. Playing {} tournaments.".format(N, tournament_count))

        next_population = population[tournament_count * 4:]

        for i in range(tournament_count):
            chromosomes = population[i * 4:(i + 1) * 4]
            winner_id = tournament(chromosomes, Player, args.games_per_tournament)
            next_population.append(chromosomes[winner_id])
            tournaments_played += 1
            print("{:.2f}: {} of {} games".format(
                tournaments_played / required_tournament_count,
                tournaments_played * args.games_per_tournament,
                required_tournament_count * args.games_per_tournament
            ))

        population = next_population
        N = len(population)

    winner = population[0]
    np.save("{}/{}.pop.winner.npy".format(folder_path, gen_id), winner)


if __name__ == '__main__':
    main()
