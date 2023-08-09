from config import global_args as g_args
from .qmix import QMixLearner
from .mappo import MAPPOLearner
from .mat import MATLearner

REGISTRY = {}
REGISTRY["qmix_learner"] = QMixLearner
REGISTRY["mappo_learner"] = MAPPOLearner
REGISTRY["mat_learner"] = MATLearner

OBJECT = {}

def get_learner(name=None):
    global REGISTRY,OBJECT
    if not name:
        name = g_args("learner")
    if name in OBJECT:
        return OBJECT[name]
    if name not in REGISTRY:
        return
    obj = REGISTRY[name]()
    OBJECT[name] = obj
    return obj

def create_learner(name=None):
    if not name:
        name = g_args("learner")
    if name not in REGISTRY:
        return
    return REGISTRY[name]()