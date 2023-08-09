# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/9/11 11:17 上午   
# 描述：选择运行环境，需要维护env/__ini__.py && config.json（存储环境默认参数）

import json
import env
import os
from config import global_args as g_args
from env.env_wrappers import DummyVecEnv,SubprocVecEnv


def make(env_type, conf=None, logdir=None):
    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not conf:
        with open(file_path) as f:
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']
    if logdir:
        return getattr(env, class_literal)(conf, logdir)
    else:
        return getattr(env, class_literal)(conf)

def make_parallel_env(dump_dir=None):
    def get_env_fn_list(rank):
        def init_env():
            env = make(g_args("env_name"), logdir=dump_dir)
            env.seed(g_args("seed"))
            return env
        return init_env
    env_fn_list = [get_env_fn_list(i) for i in range(g_args("thread_num"))]
    if g_args("thread_num") == 1:
        return DummyVecEnv(env_fn_list)
    else:
        return SubprocVecEnv(env_fn_list)


if __name__ == "__main__":
    make("classic_MountainCar-v0")
