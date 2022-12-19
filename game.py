from collections import Counter

import numpy as np

from players import *
from common import *

class Game():
    def __init__(self, chars=None, actions=None, seers=None, mc_cache=None):
        # given can contain roles, speech_order, other game-generated-rng,
        # which are directly applied; given can also contain actions,
        # which are relayed so characters report back the requested action(s)
        # with a probability.
        self.log_likelihood = 0
        self.future_actions = actions

        self.gametime = GameTime()
        self.actions = []

        self.n_players = n_players      # imported from common

        # everything below are summaries that can be parsed from actions;
        # they are used for quick heuristic evaluations by players
        self.alive = [1 for i in range(n_players)]

        self.election_order = None # [2, 6, 7, ,,,]
        self.running = [-1 for i in range(n_players)] # [0, 0, 1, ...]
        self.seer_speech = {}
        self.sheriff_votes = [-1 for i in range(n_players)]

        self.vote_history = [[-1 for i in range(n_players)] for j in range(6)]
        self.lynch_history = [[0 for i in range(n_players)] for j in range(6)]
        self.speech_history = []
        self.voted_out = []

        # role assignment can be fixed
        self.roles = rng_player_chars() if chars is None else chars
        self.roles_map = {
            char: set(i for i, c in enumerate(self.roles) if c == char)
                        for char in character_set
        }
        self.werewolves = self.roles_map[Character.WEREWOLF]
        self.goodpeople = set(range(self.n_players)) - self.werewolves
        self.seer = list(self.roles_map[Character.SEER])[0]

        if seers is None:
            self.fake_seer = np.random.choice(list(self.werewolves))
        else:
            self.fake_seer = [s for s in seers if s != self.seer][0]
        
        self.mc_cache = [] if mc_cache is None else mc_cache
        
        self.players = [char_to_class[c](i, self) for i, c in enumerate(self.roles)]
        self.players[self.fake_seer].assign_fake_seer()

        self.MCMC_pred = None

    def __repr__(self):
        s = ''
        s += 'PLAYER ROLES\n'
        s += '-'*60 + '\n'
        for i, role in enumerate(self.roles):
            if i == self.seer:
                seer_role = '\tREAL SEER'
            elif i == self.fake_seer:
                seer_role = 'FAKE SEER'
            else:
                seer_role = ''
            s += f'{i}\t{role}\t{seer_role}\n'
        s += '\n\nACTIONS\n'
        s += '-'*60 + '\n'
        s += f'running    \t{self.running}\n'
        i = 0
        while i < self.gametime.get_gametime()[0]:
            s += f'seer speech'
            for speaker, speech in self.seer_speech.items():
                s += f'\t{speaker}: {speech[i][2]} is {"bad" if speech[i][3] else "good"}'
            s += '\n'
            s += f'lynch      \t{self.lynch_history[i]}\n'
            s += f'votes       \t{self.vote_history[i+1]}\n'
            i += 1
        return s

    def append_action(self, action):
        self.actions.append(action)

        if action[1] == Action.RUN:
            self.running[action[0]] = action[2]

        if action[1] == Action.LYNCH:
            self.lynch_history.append(action)
            self.alive[action[2]] = 0

        if action[1] == Action.VOTE:
            day = self.gametime.get_gametime()[0]
            self.vote_history[day][action[0]] = action[2]
            if sum([x != -1 for x in self.vote_history[day]]) == sum(self.alive):
                vote_count = Counter(self.vote_history[day])
                _voted = max(vote_count.items(), key=lambda x:x[1])[0]
                self.voted_out.append(_voted)
                self.alive[_voted] = 0

        if action[1] == Action.ELECTION_VOTE:
            self.sheriff_votes[action[0]] = action[2]
        
        if action[1] == Action.CHECK:
            pass

        if action[1] == Action.SEER_SPEECH:
            if action[0] not in self.seer_speech:
                self.seer_speech[action[0]] = []
            self.seer_speech[action[0]].append(action)

    def run(self):
        i = 0
        while True:
            good_alive = [x for x in self.goodpeople if self.alive[x]]
            bad_alive = [x for x in self.werewolves if self.alive[x]]
            # ending condition
            if len(good_alive) == 0 or bad_alive == 0:
                return self.actions
            is_new_date, orig = self.gametime.next_phase()
            temp = self.gametime
            if is_new_date:
                self.gametime = orig
                arr, likelihood = simulate(self, N=8)
                self.mc_cache.append((arr, likelihood))
                print(self)
                print([(arr[i], likelihood[i]) for i in sorted(range(len(likelihood)), key=lambda x: likelihood[x])][:3])
                self.gametime = temp
            date, phase = self.gametime.get_gametime()
            if date > 2 or (date == 2 and phase == Phase.SEER_SPEECH):
                return
            
            if phase == Phase.WEREWOLF:
                if date == 0:
                    continue
                else:
                    rng = np.random.random()
                    if self.voted_out[-1] == self.seer:
                        if rng < 0.2:
                            self.append_action((-1, Action.LYNCH, self.fake_seer))
                            continue
                    elif self.voted_out[-1] in self.werewolves:
                        if rng < 0.7:
                            self.append_action((-1, Action.LYNCH, self.seer))
                            continue
                    self.append_action((-1, Action.LYNCH, np.random.choice(
                        [x for x in good_alive if x != self.seer])))
                    continue
            if phase == Phase.SEER:
                # in char_rng, enforce that seer must be one of the candidates
                if self.alive[self.seer]:
                    _action = self.players[self.seer].on_check()
                    self.append_action(_action)
                    i += 1
                continue
        
            if phase == Phase.ELECTION:
                # attend
                for player in self.players:
                    _action = player.on_election()
                    self.append_action(_action)
                    i += 1
                # seer speeches
                seers = sorted([self.seer, self.fake_seer])
                _action0 = self.players[seers[0]].on_seer_speech()
                self.append_action(_action0)
                _action1 = self.players[seers[1]].on_seer_speech()
                self.append_action(_action1)
                # vote
                for player_id, player in enumerate(self.players):
                    if not self.running[player_id]:
                        _action = player.on_election_vote()
                        self.append_action(_action)
                continue
        
            if phase == Phase.VOTE:
                for i, player in enumerate(self.players):
                    if self.alive[i]:
                        _action = player.on_vote()
                        self.append_action(_action)
                        i += 1

            if phase == Phase.SEER_SPEECH:
                seers = sorted([self.seer, self.fake_seer])
                _action0 = self.players[seers[0]].on_seer_speech()
                self.append_action(_action0)
                _action1 = self.players[seers[1]].on_seer_speech()
                self.append_action(_action1)

    def rerun(self):
        i = 0
        
        while i < len(self.future_actions):
            self.gametime.next_phase()
            date, phase = self.gametime.get_gametime()
            if date > 2 or (date == 2 and phase == Phase.SEER_SPEECH):
                return self.log_likelihood
            # ending condition (both nonzero length) is checked in char_rng
            good_alive = [x for x in self.goodpeople if self.alive[x]]
            bad_alive = [x for x in self.werewolves if self.alive[x]]
            
            if phase == Phase.WEREWOLF:
                if date == 0:
                    continue
                else:
                    trg_id = self.future_actions[i][2]
                    p = None
                    if trg_id in self.werewolves and trg_id != self.fake_seer:
                        p = 0
                    elif trg_id == self.fake_seer:
                        if self.voted_out[-1] == self.seer:
                            p = 0.2
                    elif trg_id == self.seer:
                        if self.voted_out[-1] == self.fake_seer:
                            p = 0.7
                    if p is None:
                        p = 1/(len([x for x in good_alive if x != self.seer]))
                    self.append_action(self.future_actions[i])
                    self.log_likelihood += np.log(p)
                    i += 1
                    continue
                        
            if phase == Phase.SEER:
                # in char_rng, enforce that seer must be one of the candidates
                if self.alive[self.seer]:
                    j = i
                    # check morning speech content
                    while j < len(self.future_actions)-1 and self.future_actions[j][1] != Action.SEER_SPEECH or self.future_actions[j][0] != self.seer:
                        j += 1
                    if self.future_actions[j][1] != Action.SEER_SPEECH:
                        return 0
                    other_id = self.future_actions[j][2]
                    side = self.future_actions[j][3]
                
                    _action = (self.seer, Action.CHECK, other_id, side)
                    self.log_likelihood += np.log(self.players[self.seer].on_check(obs=_action)[-1])
                    self.append_action(_action)
                    i += 1
                continue
        
            if phase == Phase.ELECTION:
                # can't die first day, do not casework
                for player in self.players:
                    self.log_likelihood += np.log(player.on_election(obs=self.future_actions[i])[-1])
                    self.append_action(self.future_actions[i])
                    i += 1
                for j in range(2):
                    player_id, _, other_id, side, *args = self.future_actions[i]
                    pr = self.players[player_id].on_seer_speech(obs=self.future_actions[i])[-1]
                    self.log_likelihood += np.log(pr)
                    self.append_action(self.future_actions[i])
                    i += 1
                
                for player_id, player in enumerate(self.players):
                    if not self.running[player_id]:
                        self.log_likelihood += np.log(
                            self.players[player_id].on_election_vote(obs=self.future_actions[i])[-1])
                        self.append_action(self.future_actions[i])
                        i += 1
                continue

            if phase == Phase.SEER_SPEECH:
                for j in range(2):
                    if self.future_actions[i] != Action.SEER_SPEECH:
                        continue
                    player_id, _, other_id, side, *args = self.future_actions[i]
                    self.log_likelihood += np.log(
                        self.players[player_id].on_seer_speech(obs=self.future_actions[i])[-1])
                    self.append_action(self.future_actions[i])
                    i += 1

            if phase == Phase.VOTE:
                for player_id, player in enumerate(self.players):
                    if self.alive[player_id]:
                        self.log_likelihood += np.log(
                            player.on_vote(obs=self.future_actions[i])[-1])
                        self.append_action(self.future_actions[i])
                        i += 1
                continue
                
        return self.log_likelihood


def simulate(game, N=100):
    # print(('-'*80 + '\n') * 20)
    actions = deepcopy(game.actions)
    seers = [game.seer, game.fake_seer]
    mc_cache = game.mc_cache

    arr = np.empty((2*N, 12), dtype=object)
    log_likelihood = np.empty((2*N))
    for i in range(2):
        for trial in range(N):
            roles = rng_player_chars(given={seers[i]: Character.SEER, seers[1-i]: Character.WEREWOLF})
            game = Game(actions=actions, chars=roles, seers=seers, mc_cache=mc_cache)
            game.rerun()

            arr[i*N+trial] = roles
            log_likelihood[i*N+trial] = game.log_likelihood
    
    likelihood = softmax(log_likelihood) if max(log_likelihood) > -50 else np.zeros(log_likelihood.shape)
    return arr, likelihood

# def marginal(arr, likelihood, given):
#     mask = np.ones(likelihood.shape)
#     for k, v in given.items():
#         mask = np.bitwise_and(arr[:, k] == v, mask)
#     return arr[mask], likelihood[mask]

# def marginal_pwere(arr, likelihood, given):
#     marg_arr, marg_llh = marginal(arr, likelihood, given)
#     ret = np.sum(marg_llh * (marg_arr == Character.WEREWOLF).T, axis=1) / np.sum(marg_llh)
#     # print(ret)
#     return ret