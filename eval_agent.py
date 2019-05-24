import argparse
import random
import os

import numpy as np
from progressbar import ProgressBar, Percentage

from pyludo import LudoGame, LudoPlayerRandom
from SmartPlayer import SmartPlayer
from GAPlayers import get_ga_player

fixed_players = {
    "random": LudoPlayerRandom,
    "smart": SmartPlayer,
}


def get_player(player_args):
    assert 1 <= len(player_args) <= 2
    if len(player_args) == 1:
        return fixed_players[player_args[0]]()
    Player = get_ga_player(player_args[0])
    chromosome = np.load(player_args[1])
    return Player(chromosome)


def tournament(_players, game_count):
    progress_bar = ProgressBar(widgets=[Percentage()], maxval=game_count).start()

    players = [player for player in _players]
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i
    win_rates = np.zeros(4)
    for i in range(game_count):
        random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1
        progress_bar.update(i + 1)

    progress_bar.finish()
    return win_rates / game_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", nargs="+")
    parser.add_argument("--opponent", nargs="+")
    parser.add_argument("--compare", action='store_const', const=True, default=False)
    parser.add_argument("--game_count", type=int, required=True)
    args = parser.parse_args()

    player = get_player(args.player)
    opponent = get_player(args.opponent)
    player_name = "-".join(args.player).replace("/", "-")
    opponent_name = "-".join(args.opponent).replace("/", "-")

    dist = (1, 3)
    if args.compare:
        dist = (2, 2)

    players = [player] * dist[0] + [opponent] * dist[1]
    win_rates = tournament(players, args.game_count)
    player_win_rate = np.sum(win_rates[:dist[1]])

    eval_folder_path = "agent_evaluations"
    if not os.path.isdir(eval_folder_path):
        os.mkdir(eval_folder_path)

    file_name = '{}-{}-vs-{}-{}-{:.3f}'.format(
        'compare' if args.compare else 'score', player_name, opponent_name, args.game_count, player_win_rate
    )

    f = open('{}/{}'.format(eval_folder_path, file_name), 'w')
    f.close()


if __name__ == '__main__':
    main()
