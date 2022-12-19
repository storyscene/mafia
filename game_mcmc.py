import itertools
from collections import Counter
from copy import deepcopy

import numpy as np

# from players import *
from game import *
from common import *

# ITER = 100

# # game = Game(
# #     actions=[
# #         (4, Action.CHECK, 7, 1),
# #         (0, Action.RUN, 1),
# #         (1, Action.RUN, 0),
# #         (2, Action.RUN, 1),
# #         (3, Action.RUN, 0),
# #         (4, Action.RUN, 1),
# #         (5, Action.RUN, 1),
# #         (6, Action.RUN, 0),
# #         (7, Action.RUN, 0),
# #         (8, Action.RUN, 1),
# #         (9, Action.RUN, 1),
# #         (10, Action.RUN, 0),
# #         (11, Action.RUN, 1),
# #         (4, Action.SEER_SPEECH, 7, 1),
# #         (8, Action.SEER_SPEECH, 1, 0),
# #         (1, Action.ELECTION_VOTE, 8),
# #         (3, Action.ELECTION_VOTE, 4),
# #         (6, Action.ELECTION_VOTE, 8),
# #         (7, Action.ELECTION_VOTE, 8),
# #         (10, Action.ELECTION_VOTE, 4)
# #     ],
# #     chars=[
# #         Character.TOWN,
# #         Character.WEREWOLF,
# #         Character.TOWN,
# #         Character.WEREWOLF,
# #         Character.SEER,
# #         Character.TOWN,
# #         Character.WEREWOLF,
# #         Character.TOWN,
# #         Character.WEREWOLF,
# #         Character.TOWN,
# #         Character.TOWN,
# #         Character.TOWN
# #     ],
# #     seers=[4, 8]
# # )

# def simulate(game, N=100):
#     actions = deepcopy(game.actions)
#     seers = [game.seer, game.fake_seer]

#     arr = np.empty((2*N, 12), dtype=object)
#     log_likelihood = np.empty((2*N))
#     for i in range(2):
#         for trial in range(N):
#             roles = rng_player_chars(given={seers[i]: Character.SEER, seers[1-i]: Character.WEREWOLF})
#             game = Game(actions=actions, chars=roles, seers=seers)
#             game.rerun()

#             arr[i*N+trial] = roles
#             log_likelihood[i*N+trial] = game.log_likelihood
    
#     likelihood = softmax(log_likelihood)
#     return arr, likelihood

if __name__ == '__main__':
    game = Game()
    game.run()
    print(game)

    print(game.actions)
    # new_game = Game(actions=game.actions, chars=game.roles, seers=[game.seer, game.fake_seer])
    # new_game.rerun()
    # print(new_game)
    # actions = deepcopy(game.actions)
    # seers = [game.seer, game.fake_seer]

    # arr, likelihood = simulate(game)

    # print(marginal({game.seer: Character.SEER}))