from copy import copy
from multiprocessing import Process, cpu_count
import os
import random
import torch.multiprocessing as mp
from time import sleep
import traceback

from distributed_rl.worker import Worker, Worker_multi
from distributed_rl.learner import Learner, Learner_multi
from distributed_rl.worker import Worker_grad
from distributed_rl.learner import learner_grad
from distributed_rl.mempool import MemoryPool, CreateMemPool
from log import Log, LogDebug, LogExc, LogErr
#import ray
import torch
import gfootball.env as football_env
from models import ActorCritic as net
from torch import optim, nn 
import distributed_rl.mempool as MEMPOOL


#ray.init(ignore_reinit_error=True, num_gpus=1)
g_learner = None
g_worker_manager = None



def RunJob(args, env_name):
    try:
        print(env_name+" get here!")
        slave = Worker_multi(args)
        slave.run()
        #LogDebug("worker get task, run result[%s]", result)
    except Exception as e:
        LogErr("Runjob err:%s", traceback.format_exc())

#@ray.remote
def RunJob_ray(args):
    try:
        LogDebug(args.env_name+" get here!")
        slave = Worker(args)
        result = slave.run()
    except Exception as e:
        LogErr("Runjob err:%s", traceback.format_exc())


class WorkerManager(object):
    """
    WorkerManager 负责管理多个worker的生命周期，控制worker该拉取哪个模型
    """
    def __init__(self, args, redis_args):
        """init"""
        self.args = args
        self.train_environment = self.args.env_name
        # 不能使用进程池，因为使用的一些库里面会报错，后台进程不允许有子进程
        #   File "/opt/conda/lib/python3.8/multiprocessing/process.py", line 118, in start
        # assert not _current_process._config.get('daemon'), \
        # AssertionError: daemonic processes are not allowed to have children
        # self.process_pool = Pool(processes = self.cpu_count)
        # self.process_pool.daemon = False
        self.redis_args = redis_args
        self.max_episode = self.args.nsteps
        self.redis = CreateMemPool(self.redis_args)
        LogDebug("worker manager start [%s]", redis_args)


    def start_train(self):
        mp.set_start_method('spawn') 
        LogDebug("WorkerManager pid[%s] start_train", os.getpid())
        Worker_list, process_list = [], []

        #for env_id in range(self.args.num_workers):   # 这里需要改一下，原本代码的train_enviroment是一个列表，里面8个环境现在并不是这样，现在是同一个环境，改成循环线程个数次，均创建同一个环境
        #    LogDebug("init Worker[%s]", self.train_environment+'_'+str(env_id))
        #    p = mp.Process(target=RunJob, args=(self.args, self.train_environment))
        #    p.start()
        #    process_list.append(p)
        process_list = [mp.Process(target=RunJob, args=(self.args, self.train_environment)) for env_id in range(self.args.num_workers)]
        [p.start() for p in process_list]
        [p.join() for p in process_list]
        #for env_id in range(self.args.num_workers):
        #    LogDebug("init Worker[%s]", self.train_environment+'_'+str(env_id))
        #    Worker_list.append(Worker(self.args))
        #    p = mp.Process(target=Worker_list[env_id].run)
        #    p.start()
        #    process_list.append(p)

        #for p in process_list:
        #    p.join()

        LogDebug("start_train finish")
    
    #def start_train_ray(self):
    #    Worker_list = [Worker.remote(self.args) for _ in range(self.args.num_workers)]
    #    re = ray.get([w.run.remote() for w in Worker_list])




def get_learner(args):
    global g_learner
    if not g_learner:
        g_learner = Learner_multi(args)
    return g_learner


def get_worker_manager(args, is_test=False):
    redis_config = {}
    redis_config["host"] = args.redis_ip 
    redis_config["port"] = args.redis_port
    redis_config["password"] = args.redis_pass_word 
    redis_config["test"] = is_test 

    global g_worker_manager
    if not g_worker_manager:
        g_worker_manager = WorkerManager(args, redis_config)
    return g_worker_manager

