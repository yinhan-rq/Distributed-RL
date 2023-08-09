from .object import EpisodeRunner
from .object import MultiThreadRunner

REGISTRY = {}
REGISTRY["episode_runner"] = EpisodeRunner
REGISTRY["multi_thread_runner"] = MultiThreadRunner 

def get_runner(name):
    global REGISTRY
    if name not in REGISTRY:
        return
    obj = REGISTRY[name]()
    return obj

def create_runner(name):
    global REGISTRY
    if name not in REGISTRY:
        return
    obj = REGISTRY[name]()
    return obj
