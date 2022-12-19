import numpy as np
from scipy.stats import binom
from random import choice

from common import *

class Player():
    def __init__(self, i, game):
        self._id = i
        self.game = game
        self.n_players = game.n_players
        
        self.mcmc_result = None
        self.player_visibility = [0] * game.n_players
        self.player_visibility[i] = 1

    def on_vote(self, obs=None):
        # for town and seer
        date = self.game.gametime.get_gametime()[0]
        if date > 0:
            mc_arr, mc_llh = self.game.mc_cache[date-1]
            pwere = marginal_pwere(
                mc_arr, mc_llh,
                {i: self.game.roles[i] for i in range(self.n_players)
                        if self.player_visibility[i]
                }
            )
            if obs is not None:
                return (self._id, Action.VOTE, obs[2], pwere[obs[2]])
            return (self._id, Action.VOTE,
                        np.random.choice(range(self.n_players), p=pwere))
        else:
            known_bad = [i for i in range(self.n_players)
                            if self.player_visibility[i] and i in self.game.werewolves
                            and self.game.alive[i]]
            known_good = [i for i in range(self.n_players)
                            if self.player_visibility[i] and i not in self.game.werewolves
                            and self.game.alive[i]]
            unknown = [i for i in range(self.n_players)
                            if not self.player_visibility[i] and self.game.alive[i]]
            if obs is not None:
                if len(known_bad) > 0:
                    if obs[2] in known_bad:
                        p = 1 / len(known_bad)
                    else:
                        p = 0
                else:
                    if obs[2] in known_good:
                        p = 0
                    else:
                        p = 1 / (sum(self.game.alive) - len(known_good))
                return (self._id, Action.VOTE, obs[2], p)
            if len(known_bad) > 0:
                return (self._id, Action.VOTE, np.random.choice(known_bad))
            else:
                return (self._id, Action.VOTE, np.random.choice(unknown))

class Werewolf(Player):
    def __init__(self, i, game):
        super().__init__(i, game)
        self.is_fake_seer = False
        self.friends = self.game.roles_map[Character.WEREWOLF]
        for f in self.friends:
            self.player_visibility[f] = 1

        self.town = Town(i, game)   # for acting like a town

    def assign_fake_seer(self):
        self.is_fake_seer = True
    
    def on_election(self, obs=None):
        self.real_seer = self.game.seer
        self.fake_seer = self.game.fake_seer

        if self.is_fake_seer:
            if obs is not None and obs[2] == 0:
                # not running with probability 0
                return (self._id, Action.RUN, 0, 0)
            return (self._id, Action.RUN, 1, 1)
        else:
            # basically run = flip(0.3)
            return self.town.on_election(obs)

    def on_seer_speech(self, obs=None):
        # obs is processed by game, always matches whether they are a fake seer
        if not self.is_fake_seer:
            return self.town.on_seer_speech(obs)

        valid = {i for i in range(self.n_players) if self.game.alive and i != self._id}
        if len(self.game.seer_speech) > 1:
            for claimed_before in [x[2] for x in self.game.seer_speech[self._id]]:
                print('-'*38, claimed_before)
                if claimed_before in valid:
                    valid.remove(claimed_before)
        # parse running actions so far
        running = self.game.running
        election_order = [i for i in range(self.n_players) if running[i]]
        run_ind = election_order.index(self._id)
        running_set = set(election_order)

        # prepare payoff calculation
        rem_set = set(election_order[run_ind+1:]) and valid
        rem_foes = (rem_set - self.friends) and valid
        rem_friends = (rem_set and self.friends) and valid

        prev_set = set(election_order[:run_ind]) and valid
        prev_foes = (prev_set - self.friends) and valid
        prev_friends = (prev_set and self.friends) and valid

        voting_set = (set(range(self.n_players)) - running_set) and valid
        voting_foes = (voting_set - self.friends) and valid
        voting_friends = (voting_set and self.friends) and valid
        
        if len(self.game.seer_speech) > 0:
            exists_seer = True
            seer, seer_actions = list(self.game.seer_speech.items())[0]
            self.player_visibility[seer] = 1
        else:
            exists_seer = False
        
        if exists_seer:
            if seer_actions[0][1] == 1: # found werewolf
                stakes = (
                    -GAME_PAYOFF['sheriff'],
                    -GAME_PAYOFF['werewolf']+GAME_PAYOFF['sheriff']
                )
            else:
                stakes = (-GAME_PAYOFF['sheriff'], GAME_PAYOFF['sheriff'])
        else:
            stakes = (
                -GAME_PAYOFF['sheriff'],
                -GAME_PAYOFF['werewolf']*4/11+GAME_PAYOFF['sheriff']
            )
            
        # k = allowable nonvotes, n = town votes available
        voting_prob = binom.cdf(
            k=(len(voting_set)-1)/2+len(voting_friends),
            n=len(voting_foes), p=0.5)
        standard_payoff = payoff_success_fn(p=voting_prob, stakes=stakes)
        
        strat_to_id = {
            0: self.friends,
            1: prev_foes,
            2: rem_foes,
            3: voting_foes
        }
        id_to_strat = (
            {k: 0 for k in self.friends if k != self._id} |
            {k: 1 for k in prev_foes} |
            {k: 2 for k in rem_foes} |
            {k: 3 for k in voting_foes}
        )
        # we omit werewolf accusing another werewolf because payoff hard
        strat_payoffs = {
            (0, 0): standard_payoff,
            (1, 0): standard_payoff,
            (1, 1): standard_payoff,
            (2, 0): payoff_success_fn(
                p = (voting_prob if exists_seer else
                        voting_prob * (1-1/len(rem_foes))),
                stakes=stakes
            ),
            (2, 1): payoff_success_fn(
                p = voting_prob * (1-1/self.n_players),
                stakes=stakes
            ),
            (3, 0): payoff_success_fn(
                p = binom.cdf(
                    k=(len(voting_set)-1)/2+len(voting_friends),
                    n=len(voting_foes)-1, p=0.5),
                stakes=stakes
            ),
            (3, 1): payoff_success_fn(
                p = binom.cdf(
                    k=(len(voting_set)-1)/2+len(voting_friends)-1,
                    n=len(voting_foes)-1, p=0.5),
                stakes=stakes
            )
        }
        
        for k in strat_payoffs:
            if len(strat_to_id[k[0]]) == 0:
                strat_payoffs[k] = -np.inf
        ks, vs, strat_probs = payoff_to_prob(strat_payoffs)
        # print('strat_probs:', strat_probs)

        if obs is not None:
            other_id = obs[2]
            side = obs[3]
            try:
                strat_type = id_to_strat[other_id]
                id_prob = 1 / len(strat_to_id[strat_type])
            except KeyError:
                strat_type = -1
            
            strat = (strat_type, side)
            if strat in strat_payoffs:
                strat_prob = strat_probs[strat]
                return (self._id, Action.SEER_SPEECH,
                        other_id, side, strat_prob*id_prob)
            else:
                return (self._id, Action.SEER_SPEECH,
                        other_id, side, 0)
        else:
            strat_id = np.random.choice(range(len(ks)), p=vs)
            strat_type, side = ks[strat_id]
            other_id = np.random.choice(list(strat_to_id[strat_type]))
            return (self._id, Action.SEER_SPEECH, other_id, side, 1)

    def on_election_vote(self, obs=None):
        if obs is not None:
            if obs[2] == self.real_seer:
                return (*obs[:3], 0.1)
            else:
                return (*obs[:3], 0.9)
        rng = np.random.random()
        if rng < 0.1:
            return (self._id, Action.ELECTION_VOTE, self.real_seer, 1)
        else:
            return (self._id, Action.ELECTION_VOTE, self.fake_seer, 1)

    def on_vote(self, obs=None):
        date = self.game.gametime.get_gametime()[0]
        if date > 0:
            mc_arr, mc_llh = self.game.mc_cache[date-1]
            # assume they are good themselves
            pwere = marginal_pwere(
                mc_arr, mc_llh,
                {self._id: Character.SEER if self.is_fake_seer else Character.TOWN}
            )
            # try not to vote for friends
            for f in self.friends:
                pwere[f] /= 3
            pwere /= sum(pwere)
            if obs is not None:
                return (self._id, Action.VOTE, obs[2], pwere[obs[2]])
            return (self._id, Action.VOTE,
                        np.random.choice(range(self.n_players), p=pwere))
        else:
            speech = self.game.seer_speech[self.fake_seer][-1]
            rng = np.random.random()
            if speech[3] == 1:
                if obs is not None:
                    if obs[2] == speech[2]:
                        return (self._id, Action.VOTE, obs[2], 0.8)
                    else:
                        return (self._id, Action.VOTE, obs[2], 0.2/7)
                if rng < 0.8:
                    return (self._id, Action.VOTE, speech[2])
                return (self._id, Action.VOTE,
                    np.random.choice(
                        [i for i in range(n_players) if i not in self.friends and i != self.real_seer]))
            else:
                if obs is not None:
                    if obs[2] == self.real_seer:
                        return (self._id, Action.VOTE, obs[2], 0.8)
                    else:
                        return (self._id, Action.VOTE, obs[2], 0.2/7)
                if rng < 0.8:
                    return (self._id, Action.VOTE, self.real_seer)
                else:
                    return (self._id, Action.VOTE,
                        np.random.choice(
                            [i for i in range(n_players) if i not in self.friends and i != self.real_seer]))

class Town(Player):
    def __init__(self, i, game):
        super().__init__(i, game)
        self.known_real_seer = None
        self.known_fake_seer = None

    def on_election(self, obs=None):
        # basically run = flip(0.3)
        if obs is not None:
            if obs[2] == 0:
                return (self._id, Action.RUN, 0, 0.3)
            else:
                return (self._id, Action.RUN, 1, 0.7)
        rng = np.random.random()
        if rng < 0.3:
            return (self._id, Action.RUN, 0, 1)
        else:
            return (self._id, Action.RUN, 1, 1)

    def on_seer_speech(self, obs=None):
        if obs is not None:
            return (*obs[:4], 0)
        return None

    def on_election_vote(self, obs=None):
        speeches = self.game.seer_speech
        seers = []
        cdn_speeches = []
        for seer in speeches:
            seers.append(seer)
            cdn_speeches.append((seer, speeches[seer][0][2], speeches[seer][0][3]))
        p = 0.5 # prob picking seer 0
        for i in range(2):
            if cdn_speeches[i][1] == self._id:
                if cdn_speeches[i][2] == 1:
                    self.player_visibility[seers[0]] = 1
                    self.player_visibility[seers[1]] = 1
                    p = 0 if i == 0 else 1
                else:
                    # result: 0.9 or 0.1 if praised, 0.5 if praised twice
                    p += 0.4 * (1-2*i)
        if obs is not None:
            return (*obs, p if obs[2] == seers[0] else 1-p)
        rng = np.random.random()
        if rng < p:
            return (self._id, Action.ELECTION_VOTE, seers[0], 1)
        else:
            return (self._id, Action.ELECTION_VOTE, seers[1], 1)

class Seer(Player):
    def __init__(self, i, game):
        super().__init__(i, game)
        self.check_seq = []

    def on_check(self, obs=None):
        print(self.player_visibility)
        alive = self.game.alive
        rem_unknown = [i for i in range(self.n_players)
                            if alive[i] and not self.player_visibility[i]]
        p = 1 / len(rem_unknown)
        if obs is not None:
            _, _, checked_id, side, *args = obs
            if (self.game.roles[checked_id] == Character.WEREWOLF) != side:
                p = 0
            self.player_visibility[checked_id] = 1
            self.check_seq.append(obs)
            return (self._id, Action.CHECK, checked_id, side, p)
        else:
            checked_id = np.random.choice(rem_unknown)
            side = 0 if self.game.roles[checked_id] == Character.TOWN else 1
            self.player_visibility[checked_id] = 1
            _action = (self._id, Action.CHECK, checked_id, side, 1)
            self.check_seq.append(_action)
            return _action

    def on_election(self, obs=None):
        if obs is not None:
            if obs[2] == 0:
                return (self._id, Action.RUN, 0, 0)
            else:
                return (self._id, Action.RUN, 1, 1)
        return (self._id, Action.RUN, 1, 1)

    def on_seer_speech(self, obs=None):
        # observation should be verified in on_check
        last_check = self.check_seq[-1]
        return (self._id, Action.SEER_SPEECH, last_check[2], last_check[3], 1)

char_to_class = {
    Character.WEREWOLF: Werewolf,
    Character.SEER: Seer,
    Character.TOWN: Town
}