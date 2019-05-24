import numpy as np
from pyludo import LudoState, LudoStateFull
from pyludo.utils import token_vulnerability


class GABasePlayer:
    name = "base"
    gene_count = None

    def __init__(self, chromosome):
        self.chromosome = chromosome

    def play(self, state, dice_roll, next_states):
        full_state = LudoStateFull(state, dice_roll, next_states)
        action_values = self.eval_actions(full_state)
        actions_prioritized = np.argsort(-action_values)
        for token_id in actions_prioritized:
            if next_states[token_id] is not False:
                return token_id

    def eval_actions(self, full_state: LudoStateFull):
        pass

    @staticmethod
    def normalize(chromosome):
        return chromosome

    def pop_init(self, sigma_count, pop_size):
        genes = np.random.randn(pop_size * self.gene_count).reshape((pop_size, -1))
        sigma = np.exp(np.random.randn(pop_size * sigma_count) - 2).reshape((pop_size, -1))
        return np.concatenate((genes, sigma), axis=1)


class GASimplePlayer(GABasePlayer):
    name = "simple"
    args = []
    gene_count = 4

    def __init__(self, chromosome):
        super(GASimplePlayer, self).__init__(chromosome)

    @staticmethod
    def count_home_tokens(opponents):
        home = 0
        for opponent in opponents:
            for token in opponent:
                if token == -1:
                    home += 1
        return home

    def eval_action(self, state, next_state, token_id):
        if next_state == False:
            return 0

        cur_token_pos = state[0][token_id]
        next_token_pos = next_state[0][token_id]
        cur_opponents = state[1:]
        next_opponents = next_state[1:]

        moved_out = next_token_pos > -1 == cur_token_pos
        enter_goal = next_token_pos == 99 > cur_token_pos
        enter_safe_zone = next_token_pos > 51 >= cur_token_pos
        opps_hit_home = self.count_home_tokens(next_opponents) - self.count_home_tokens(cur_opponents)
        reduced_state = [moved_out, enter_goal, enter_safe_zone, opps_hit_home]
        return sum([gene * val for gene, val in zip(self.chromosome, reduced_state)])

    def eval_actions(self, full_state: LudoStateFull):
        action_scores = np.empty(4)
        for i in range(4):
            action_scores[i] = self.eval_action(full_state.state, full_state.next_states[i], i)
        return action_scores

    @staticmethod
    def normalize(chromosome):
        gene_count = GASimplePlayer.gene_count
        chromosome[:gene_count] /= np.abs(chromosome[:gene_count]).sum() * 0.25
        return chromosome


class GAAdvancedPlayer(GABasePlayer):
    name = "advanced"
    args = []
    gene_count = 7

    def __init__(self, chromosome):
        super(GAAdvancedPlayer, self).__init__(chromosome)

    @staticmethod
    def token_progress_potential(token, params):
        if token == -1:
            return 0
        if token < 52:
            return params[0] + token / 51 * params[1]
        if token < 99:
            return params[0] + params[1] + params[2]
        return params[0] + params[1] + params[2] + params[3]

    def eval_actions(self, full_state: LudoStateFull):
        action_scores = np.zeros(4)
        for action_id, state in enumerate(full_state.next_states):
            if state == False:
                action_scores[action_id] = -1e9
                continue

            token_potentials = np.empty((4, 4))
            for player_id in range(4):
                relative_state = LudoState.get_state_relative_to_player(state, player_id)
                for token_id in range(4):
                    token = relative_state[0][token_id]
                    token_prog = self.token_progress_potential(token, self.chromosome[:4])
                    token_vuln = token_vulnerability(relative_state, token_id)
                    approx_stay_probability = (5 / 6) ** token_vuln
                    token_pot = token_prog * approx_stay_probability
                    token_potentials[player_id, token_id] = token_pot

            player_potentials = token_potentials.mean(axis=1)
            player_potential = player_potentials[0]
            opponents_rank = np.argsort(-player_potentials[1:]) + 1
            opponent_potentials = player_potentials[opponents_rank]
            action_scores[action_id] = player_potential - np.sum(opponent_potentials * self.chromosome[4:7])
        return action_scores


class GAFullPlayer(GABasePlayer):
    name = "full"
    args = []
    inp_size = 4 * 59 + 1
    hidden_size = 100
    gene_count = (4 * 59 + 1) * 100 + 100

    def __init__(self, chromosome):
        super(GAFullPlayer, self).__init__(chromosome)
        w0_len = self.inp_size * self.hidden_size
        w1_len = self.hidden_size
        self.w0 = chromosome[:w0_len].reshape(self.inp_size, self.hidden_size)
        self.w1 = chromosome[w0_len:w0_len + w1_len].reshape(self.hidden_size)

    def eval_actions(self, full_state: LudoStateFull):
        action_scores = np.zeros(4)
        for action_id, state in enumerate(full_state.next_states):
            if state == False:
                action_scores[action_id] = -1e9
                continue
            flat_state_rep = np.zeros(self.inp_size)
            flat_state_rep[-1] = 1  # bias
            full_state_rep = flat_state_rep[:self.inp_size - 1].reshape((4, 59))
            for player_id in range(4):
                for token in state[player_id]:
                    full_state_rep[player_id][min(token + 1, 58)] += 1
            hidden = np.tanh((flat_state_rep @ self.w0) * np.sqrt(1 / self.inp_size))
            out = hidden @ self.w1
            action_scores[action_id] = out
        return action_scores


def get_ga_player(name):
    players = [GASimplePlayer, GAAdvancedPlayer, GAFullPlayer]
    player_map = {}
    for player in players:
        player_map[player.name] = player
    return player_map[name]
