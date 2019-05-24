import glob
import numpy as np
import os
from GAPlayers import get_ga_player
from pyludo import LudoPlayerRandom, LudoPlayerDefensive
from SmartPlayer import SmartPlayer


def get_player_class(folder_path):
    player_name, *player_args = os.path.basename(folder_path).split("+")[0].split("-")
    Player = get_ga_player(player_name)
    player_args = [arg_def[1](arg) for arg, arg_def in zip(player_args, Player.args)]
    return Player, player_args


def get_opponent_class(opponent_name):
    Opponents = [LudoPlayerRandom, LudoPlayerDefensive, SmartPlayer]
    opp_map = {}
    for opp in Opponents:
        opp_map[opp.name] = opp
    return opp_map[opponent_name]


def load_scores(folder_path):
    assert os.path.isdir(folder_path), "no folder found: {}".format(folder_path)
    score_paths = glob.glob(folder_path + "/*.scores.*.npy")
    X = np.array([int(os.path.basename(score_path).split(".")[0]) for score_path in score_paths])
    Y = np.array([np.load(f) for f in score_paths])
    idx = np.argsort(X)
    return X[idx], Y[idx]


def load_populations(folder_path):
    assert os.path.isdir(folder_path), "no folder found: {}".format(folder_path)
    population_paths = glob.glob(folder_path + "/*.pop.npy")
    generation_ids = np.array([int(os.path.basename(path).split(".")[0]) for path in population_paths])
    populations = np.array([np.load(f) for f in population_paths])
    idx = np.argsort(generation_ids)
    generation_ids = generation_ids[idx]
    populations = populations[idx]
    Player, _ = get_player_class(folder_path)
    gene_count = Player.gene_count
    genes = populations[:, :, :gene_count]
    sigmas = populations[:, :, gene_count:]
    return generation_ids, genes, sigmas
