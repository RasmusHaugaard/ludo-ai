import os
import multiprocessing as mp
import random
from glob import glob
import argparse

import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pyludo import LudoGame
from GAPlayers import get_ga_player
from ga_utils import get_opponent_class


def get_score_file_name(generation_id, opponent_name):
    return "{}.scores.{}.npy".format(generation_id, opponent_name)


def get_score_file_path(folder_path, generation_id, opponent_name):
    return "{}/{}".format(folder_path, get_score_file_name(generation_id, opponent_name))


def eval_population_worker(queue: mp.Queue, games_per_chromosome, task_counter_queue: mp.Queue, Opponent):
    while True:
        population_path = queue.get()
        folder_path = os.path.dirname(population_path)
        folder_name = os.path.basename(folder_path)

        player_name = folder_name.split("+")[0]
        Player = get_ga_player(player_name)

        population = np.load(population_path)
        N = min(len(population), 20)
        population_idx = np.random.choice(np.arange(len(population)), N, replace=False)
        population = population[population_idx]
        save_matrix = np.empty((2, N), np.float)
        save_matrix[0] = population_idx
        scores = save_matrix[1]
        for i, chromosome in enumerate(population):
            players = [Player(chromosome)] + [Opponent() for _ in range(3)]
            win_count = 0
            for _ in range(games_per_chromosome):
                random.shuffle(players)
                game = LudoGame(players)
                winner = players[game.play_full_game()]
                if isinstance(winner, Player):
                    win_count += 1
            scores[i] = win_count / games_per_chromosome

        generation_str = os.path.basename(population_path).split(".")[0]
        scores_path = get_score_file_path(folder_path, generation_str, Opponent.name)
        assert not os.path.exists(scores_path), "Scores already exists: {}".format(scores_path)
        np.save(scores_path, save_matrix)

        task_counter_queue.put(('finished', population_path))


def handle_file_path_worker(path_queue: mp.Queue, population_queue: mp.Queue, task_counter_queue: mp.Queue, Opponent):
    files_processed = set()
    while True:
        file_path = path_queue.get()
        if file_path in files_processed:
            continue
        file_name = os.path.basename(file_path)
        name_parts = file_name.split(".")
        if len(name_parts) != 3 or name_parts[1] != "pop":
            continue
        generation_str = name_parts[0]
        folder_path = os.path.dirname(file_path)
        if os.path.exists(get_score_file_path(folder_path, generation_str, Opponent.name)):
            files_processed.add(file_path)
            continue
        task_counter_queue.put(('starting', file_path))
        population_queue.put(file_path)


class FileCreatedHandler(FileSystemEventHandler):
    def __init__(self, queue: mp.Queue):
        self.queue = queue

    def on_created(self, event):
        self.queue.put(event.src_path)

    def on_moved(self, event):
        self.queue.put(event.dest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--games_per_chromosome", type=int, required=True)
    parser.add_argument("--process_count", type=int, required=True)
    parser.add_argument("--opponent_name", type=str, required=True)
    args = parser.parse_args()

    path = args.path
    assert os.path.isdir(args.path)
    games_per_chromosome = args.games_per_chromosome
    process_count = args.process_count
    opponent_name = args.opponent_name
    Opponent = get_opponent_class(opponent_name)

    path_queue = mp.Queue()
    population_queue = mp.Queue()
    task_counter_queue = mp.Queue()

    path_worker = mp.Process(target=handle_file_path_worker,
                             args=(path_queue, population_queue, task_counter_queue, Opponent))
    path_worker.start()

    pool = mp.Pool(process_count, eval_population_worker,
                   (population_queue, games_per_chromosome, task_counter_queue, Opponent))

    observer = Observer()
    observer.schedule(FileCreatedHandler(path_queue), path=path, recursive=True)
    observer.start()

    for file_path in glob(path + "/**/*.pop.npy", recursive=True):
        path_queue.put(file_path)

    unfinished_tasks = 0

    print("watching folder '{}' for new populations to evaluate...".format(path))

    try:
        while True:
            action, pop_path = task_counter_queue.get()
            if action == 'starting':
                unfinished_tasks += 1
            elif action == 'finished':
                unfinished_tasks -= 1
            print("pending tasks:", unfinished_tasks, action, pop_path)
    except KeyboardInterrupt:
        pass

    print("stopping processes")
    path_worker.terminate()
    observer.stop()
    pool.close()

    path_worker.join()
    print("path worker stopped")
    observer.join()
    print("observer stopped")
    pool.join()
    print("worker processes stopped")


if __name__ == "__main__":
    main()
