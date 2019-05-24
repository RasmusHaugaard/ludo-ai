import os
import argparse
import glob
import functools
import sys

import numpy as np

from Selections import get_selection
from Recombinators import get_recombinator
from Mutators import get_mutator
from GAPlayers import get_ga_player


def parse_args(args, required_args):
    assert len(args) == len(required_args), 'expexted args: {}'.format(required_args)
    parsed_args = []
    str_args = []
    for arg_i, (arg_name, typ) in enumerate(required_args):
        provided_arg_name, val = args[arg_i].split("=")
        assert provided_arg_name == arg_name, 'expected "{}", got "{}"'.format(arg_name, provided_arg_name)
        parsed_args.append(typ(val))
        str_args.append(val)
    return parsed_args, str_args


def args_str_to_string(args_str):
    if args_str:
        return "-" + "-".join(args_str)
    return ""


def save(folder_path, gen_id, population):
    file_writing_name = folder_path + "/{}.pop.writing.npy".format(gen_id)
    file_written_name = folder_path + "/{}.pop.npy".format(gen_id)
    np.save(file_writing_name, population)
    os.rename(file_writing_name, file_written_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", nargs='+', required=True)
    parser.add_argument("--selection", nargs='+', required=True)
    parser.add_argument("--recombination", nargs='+', required=True)
    parser.add_argument("--mutation", nargs='+', required=True)
    parser.add_argument("--gen_count", type=int, required=True)
    parser.add_argument("--save_nth_gen", type=int, required=True)
    parser.add_argument("--cont", action="store_const", const=True, default=False)
    args = parser.parse_args()

    Player = get_ga_player(args.player[0])
    player_args, player_args_str = parse_args(args.player[1:], Player.args)
    gene_count = Player.gene_count

    Selection = get_selection(args.selection[0])
    selection_args, selection_args_str = parse_args(args.selection[1:], Selection.args)

    Recombinator = get_recombinator(args.recombination[0])
    recombinator_args, recombinator_args_str = parse_args(args.recombination[1:], Recombinator.args)

    Mutator = get_mutator(args.mutation[0])
    mutator_args, mutator_args_str = parse_args(args.mutation[1:], Mutator.args)

    mutator = Mutator(gene_count, *mutator_args)
    recombinator = Recombinator(gene_count, *recombinator_args)

    generation_count = args.gen_count
    save_every_nth_generation = args.save_nth_gen

    pop_init = functools.partial(Player.pop_init, Player, mutator.chromosome_length - Player.gene_count)
    selection = Selection(Player, pop_init, recombinator, mutator, *selection_args)

    folder_name = "{}{}+{}{}+{}{}+{}{}".format(
        Player.name, args_str_to_string(player_args_str),
        Selection.name, args_str_to_string(selection_args_str),
        Recombinator.name, args_str_to_string(recombinator_args_str),
        Mutator.name, args_str_to_string(mutator_args_str),
    )
    folder_path = "populations/" + folder_name

    assert os.path.isdir(folder_path) == args.cont, '{} should{} exist'.format(folder_path, '' if args.cont else ' not')
    if not args.cont:
        os.mkdir(folder_path)
    else:
        gen_ids = [int(os.path.basename(path).split(".")[0]) for path in glob.glob(folder_path + "/*.pop.npy")]
        selection.current_generation = max(gen_ids)
        selection.population = np.load(folder_path + "/{}.pop.npy".format(selection.current_generation))

    if generation_count == 0:
        generation_count = int(1e9)

    if not args.cont:
        save(folder_path, 0, selection.get_flat_pop())
    for i in range(selection.current_generation, generation_count):
        selection.step()
        if selection.current_generation % save_every_nth_generation == 0:
            save(folder_path, selection.current_generation, selection.get_flat_pop())
        flat_pop = selection.get_flat_pop()
        chromo_mean = flat_pop.mean(axis=0)
        chromo_std = flat_pop.std(axis=0)
        print("gene mean", chromo_mean[:gene_count])
        print("gene std ", chromo_std[:gene_count])
        print("sigma mean", chromo_mean[gene_count:])
        print("sigma std ", chromo_std[gene_count:])
        print(*sys.argv[1:])

if __name__ == '__main__':
    main()
