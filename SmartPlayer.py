"""
Credit: Christian Quist Nielsen
"""
import random


class SmartPlayer:
    name = "smart"

    @staticmethod
    def play(state, dice_roll, next_states):
        if dice_roll == 6:
            home_players = [i for i, token in enumerate(state[0]) if token == -1]
            if home_players:
                return random.choice(home_players)

        max_val = -1e9
        max_i = 0
        for i, state in enumerate(next_states):
            if state != False:
                val = sum(state[0]) - sum([sum(tokens) for tokens in state[1:]])
                if val > max_val:
                    max_val = val
                    max_i = i
        return max_i
