from enum import Enum
import numpy as np
from copy import deepcopy

class Character(Enum):
    TOWN = 0
    SEER = 10
    DOCTOR = 11
    GUARD = 12      # 守卫
    HUNTER = 13     # 猎人
    BLANK = 14      # 白痴
    WEREWOLF = 20

class Phase(Enum):
    # night
    WEREWOLF = 0
    SEER = 1
    DOCTOR = 2
    GUARD = 3
    HUNTER = 4
    # day
    ELECTION = 10
    SEER_SPEECH = 11
    DISCUSSION = 12
    VOTE = 13

phase_order = [
    Phase.WEREWOLF,
    Phase.SEER,
    Phase.ELECTION,
    Phase.SEER_SPEECH,
    Phase.VOTE
]

class Action(Enum):
    LYNCH = 0   # -1, lynch, other_id
    VOTE = 1    # id, vote, other_id
    CHECK = 2   # id, check, other_id, side
    RUN = 3
    ELECTION_VOTE = 4   # id, election_vote, other_id
    SEER_SPEECH = 5     # id, seer_speech, other_id, side
    OPINION = 6 # id, opinion, other_id, side

class GameTime():
    def __init__(self, date=-1, phase=0):
        self.date = date
        self.phase = phase
    
    def next_phase(self):
        orig_gametime = GameTime(self.date, self.phase)
        if self.date == -1:
            self.date = 0
            self.phase = 0
            return False, orig_gametime
        self.date += (self.phase + 1) // len(phase_order)
        self.phase = (self.phase + 1) % len(phase_order)
        if phase_order[self.phase] == Phase.ELECTION and self.date != 0:
            self.next_phase()
        if phase_order[self.phase] == Phase.SEER_SPEECH and self.date == 0:
            self.next_phase()
        return self.date > orig_gametime.date, orig_gametime

    def get_gametime(self):
        return (self.date, phase_order[self.phase])


def multinomial(lst):
    # cr stackoverflow
    res = 1
    i = sum(lst)
    for a in lst:
        for j in range(1,a+1):
            res *= i
            res //= j
            i += 1
    return res

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def payoff_to_prob(payoffs):
    ks, vs = zip(*payoffs.items())
    vs = softmax(vs)
    probs_dict = dict(zip(ks, vs))
    return ks, vs, probs_dict


# Define the number of players and the number of mafia players
n_players = 12
character_set = {
    Character.WEREWOLF: 4,
    Character.TOWN: 7,
    Character.SEER: 1
}
character_list = [[c] * v for c, v in character_set.items()]
character_list = [i for j in character_list for i in j]     # flatten

n_permutations = multinomial(list(character_set.values()))  # 831600

def rng_player_chars(given=None):
    if given is None:
        return np.random.permutation(character_list)

    ret = [None] * n_players
    cset_copy = deepcopy(character_set)
    for k, v in given.items():
        cset_copy[v] -= 1
        ret[k] = v
    clist_copy = [[c] * v for c, v in cset_copy.items()]
    clist_copy = [i for j in clist_copy for i in j]
    clist_copy = np.random.permutation(clist_copy)
    rem_indices = [i for i in range(n_players) if i not in given]
    for i, j in enumerate(rem_indices):
        ret[j] = clist_copy[i]
    return ret

def rng_swap(roles, given=None):
    # mutates
    indices = [i for i in range(n_players) if i not in given]
    while True:
        a, b = np.random.choice(indices, size=2)
        if roles[a] != roles[b]:
            temp = roles[a]
            roles[a] = roles[b]
            roles[b] = temp
            return roles

def marginal(arr, likelihood, given):
    mask = np.ones(likelihood.shape)
    for k, v in given.items():
        mask = np.logical_and(arr[:, k] == v, mask)
    return arr[mask], likelihood[mask]

def marginal_pwere(arr, likelihood, given):
    marg_arr, marg_llh = marginal(arr, likelihood, given)
    ret = np.sum(marg_llh * (marg_arr == Character.WEREWOLF).T, axis=1) / max(1e-10, np.sum(marg_llh))
    # print(ret)
    return ret / sum(ret)
    # ret = np.clip(ret, a_min=1e-10, a_max=max(ret))
    # ret /= sum(ret)
    return ret

# STRAT_PARAM = {
#     Character.WEREWOLF: {
#         # assume one fake seer for simplicity
#         'seer': {
#             'claim': {
#                 (0, True): 0.25,    # non election town & claim good
#                 (0, False): 0.05,   # non election town & claim bad
#                 (1, True): 0.3,     # election town & claim good
#                 (1, False): 0.2,    # election town & claim bad
#                 (2, True): 0.15,    # mafia & claim good
#                 (2, False): 0.05    # mafia & claim bad
#             }
#         }
#     },
#     Character.DOCTOR: {
#         'poison': {
#             'correct_EV': 1
#         }
#     }
# }

GAME_PAYOFF = {
    'seer': 4,
    'town': 1,
    'werewolf': -3,
    'sheriff': 2
}

def payoff_success_fn(p, stakes):
    return p * stakes[0] + (1-p) * stakes[1]