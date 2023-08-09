# -*- coding:utf-8  -*-
import argparse
import alphastar
import torch
from trainer import get_trainer
from config import init_global_args

def start_training(my_alg, opponent, start_from=None, is_test=False):
    print("开始训练")
    trainer = get_trainer(my_alg, opponent) 
    if start_from:
        trainer.load_from(start_from)
    if is_test:
        trainer.test()
    else:
        trainer.train()

def start_gail(my_alg, opponent, scene):
    # 开启模仿学习阶段
    init_global_args(my_alg, scene)
    trainer = get_trainer(my_alg, opponent) 
    trainer.train()
    return trainer.get_model()

def get_models_from_path(path, lst_name):
    d = {}
    for name in lst_name:
        d[name] = torch.load(f"./models/{path}/{name}.pt")
    return d

def start_self_play(config_file, model=None):
    # 开启自博弈阶段
    init_global_args(config_file)
    alphastar.start(model)

def test_simple_training(config_file, model=None):
    # 单独启动一次训练，没有自博弈，没有模仿学习
    init_global_args(config_file)
    print("开始训练")
    trainer = get_trainer("mappo", "random") 
    print(trainer)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_alg", default="mappo", help="qmix/mappo")
    parser.add_argument("--my_scene", default="test", help= "sysu_3_vs_2/P&S_with_keeper/football_11_vs_11_sysu")
    parser.add_argument("--opponent", default="random", help="football_11v11_mappo/random/buildin_ai")
    parser.add_argument("--load", default="")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    print("load from", args.load, args.test)
    if args.load:
        lst_name = ("actor", "critic")
        start_from = get_models_from_path(args.load, lst_name)
    # start_training(args.my_alg, args.opponent, args.load, args.test)
    # start_gail(args.my_alg, args.opponent, args.my_scene)
    start_self_play("unit_test", model=start_from)
    # test_simple_training("transformer")
    # test_simple_training("mappo_test")


