# -*- coding: utf-8 -*-
import pickle
from time import sleep
import datetime
import redis
import os

from log import Log, LogDebug, LogExc, LogErr


def CreateMemPool(config):
    return MemoryPool(config)


class Model(object):
    def __init__(self, data):
        self.data = data
        pass


class MemoryPool(object):
    """
    存放worker训练得到的数据， 供worker写入， 供Learner拉取
    样本数据：多woker写入，单learner读取
    模型：单learner写入，多worker读取
    """
    def __init__(self, config):
        self.config = config
        self.pool = redis.ConnectionPool(
            host=config["host"], password=config["password"], 
            port=config.get("port", 6379))
        self.prefix = "yh_cartpole_football_" if config.get("test") else "test_cartpole_"
        self.model_prefix = "_MODEL_"
        self.sample_prefix = "SAMPLE"
        self.model_data_path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
        self.buffer_size = 500
        self.last_index = 0
        if not os.path.exists(self.model_data_path):
            os.makedirs(self.model_data_path)
            LogDebug("create dir:%s", self.model_data_path)

    def __get_model_name_key(self, model_name):
        return self.prefix + self.model_prefix + model_name
    
    def __get_sample_name_key(self):
        return self.prefix + self.sample_prefix
    
    def __get_model_list_key(self, env):
        return self.prefix + env + '_model_list'

    def push_model(self, env_id, model_name, model_obj):
        r = redis.Redis(connection_pool=self.pool)
        data = pickle.dumps(model_obj)
        r.set(self.__get_model_name_key(f"{model_name}"), data)
        r.lpush(self.__get_model_list_key(str(env_id)), model_name)
        LogDebug(f"push {env_id} {model_name}")

    def pull_model(self, env_id, model_name):
        r = redis.Redis(connection_pool=self.pool)
        model_data = r.get(self.__get_model_name_key(f"{model_name}"))
        if not model_data:
            return
        model_obj = pickle.loads(model_data)
        # save data to local disk, "b" mean bytes
        # with open(save_path, "ab") as fw:
            # fw.write(model_data)
            # LogDebug("save model[%s] to path:[%s]", model_name, save_path)
        return model_obj

    def push_sample(self, sample_info):
        r = redis.Redis(connection_pool=self.pool)
        data = pickle.dumps(sample_info)
        r.lpush(self.__get_sample_name_key(), data)
        if r.llen(self.__get_sample_name_key()) > self.buffer_size:
            r.rpop(self.__get_sample_name_key())
        #LogDebug("push sample:%s %s", self.__get_sample_name_key(), sample_info.keys)

    def pull_sample(self):
        """每次只读一个"""
        r = redis.Redis(connection_pool=self.pool)
        # 在name对应的列表的左侧获取第一个元素并在列表中移除，返回值则是第一个元素
        #while True:
        #    sample_info = r.lpop(self.__get_sample_name_key())
        #    if sample_info:
        #        break
        #    sleep(2)
        #    LogDebug("waiting samples")
        sample_info = r.blpop(self.__get_sample_name_key(), 0)
        data = pickle.loads(sample_info[1])
        return data

    def get_model_list(self, env_id):
        r = redis.Redis(connection_pool=self.pool)
        model_list = r.lrange(self.__get_model_list_key(env_id), 0, -1)
        model_list = [i.decode('utf8') for i in model_list]
        return model_list

    def get_sample_list_len(self):
        r = redis.Redis(connection_pool=self.pool)
        l = r.llen(self.__get_sample_name_key())
        return l

    def get_grad_list_len(self, model_name):
        r = redis.Redis(connection_pool=self.pool)
        l = r.llen(self.__get_model_list_key(model_name)+"_grad")
        return l
    
    def push_grad(self, data, model_name):
        r = redis.Redis(connection_pool=self.pool)
        data = pickle.dumps(data)
        r.lpush(self.__get_model_list_key(model_name)+"_grad", data)
        LogDebug(f"push model grad")

    def pull_grad(self,model_name):
        r = redis.Redis(connection_pool=self.pool)

        while True:
            grad_info = r.lpop(self.__get_model_list_key(model_name)+"_grad")
            if grad_info:
                break
            sleep(3)
            LogDebug("waiting grads")
        data = pickle.loads(grad_info)
        return data