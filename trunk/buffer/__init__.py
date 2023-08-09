from .object import ReplayBuffer
from .object import SharedReplayBuffer
from .object import ExpertBuffer

REGISTRY = {}
REGISTRY["expert_buffer"] = ExpertBuffer

OBJECT = {}

def get_buffer(name=None):
    global REGISTRY,OBJECT
    if name in OBJECT:
        return OBJECT[name]
    if name not in REGISTRY:
        return
    obj = REGISTRY[name]()
    OBJECT[name] = obj
    return obj
