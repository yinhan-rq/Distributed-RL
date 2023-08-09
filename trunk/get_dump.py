import argparse
import torch
import numpy as np
from env.chooseenv import make, make_parallel_env
from config import global_args as g_args
from learner import get_learner
from config import init_global_args

def get_models_from_path(path=None):
    if path:
        d = {}
        d["actor"] = torch.load(f"./models/mappo/{path}/actor.pt", map_location='cpu') #本地用cpu跑
        d["critic"] = torch.load(f"./models/mappo/{path}/critic.pt", map_location='cpu')
        return d

def step(env, actions, opp_actions):
    all_actions = np.squeeze(actions, axis=-1).tolist()
    opp_actions = np.squeeze(opp_actions, axis=-1).tolist()
    all_actions[0] = all_actions[0] + opp_actions[0]
    obs, _, _, _, _, _ = env.step(all_actions)
    return obs

def run_game(game, path1, path2):
    player1 = get_learner()
    player1.load_dict(get_models_from_path(path1))
    player2 = get_learner()
    player2.load_dict(get_models_from_path(path2))

    obs, _ = game.reset()
    for st in range(g_args("eval_episode_length")):
        if st % 100 == 0:
            print("step:", st)

        obs1 = obs[:, 0:g_args("left_agent_num"), :]
        available_actions1 = np.array(obs1[...,: game.action_space.n])
        obs1 = np.concatenate(obs1)
        actions1, _ = player1.get_actions(obs1, available_actions1)

        obs2 = obs[:, g_args("left_agent_num"):, :]
        available_actions2 = np.array(obs2[...,: game.action_space.n])
        obs2 = np.concatenate(obs2)
        actions2, _ = player2.get_actions(obs2, available_actions2)

        obs = step(game, actions1, actions2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_alg", default="mappo", help="qmix/mappo")
    parser.add_argument("--load1", default="")
    parser.add_argument("--load2", default="")
    args = parser.parse_args()
    init_global_args(args.my_alg, 'vs')
    logdir = "./IL_test/" + args.load1.replace("/", "_") + "vs" + args.load2.replace("/", "_")
    game = make_parallel_env(dump_dir=logdir)
    run_game(game, args.load1, args.load2)

