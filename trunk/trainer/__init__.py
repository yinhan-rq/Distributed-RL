from config import global_args as g_args
from .qmix_trainer import QMixTrainer
from .mappo_trainer import MAPPOTrainer

REGISTRY = {}
REGISTRY["qmix"] = QMixTrainer
REGISTRY["mappo"] = MAPPOTrainer

def get_trainer(name, opponent):
    global REGISTRY
    if name not in REGISTRY:
        return
    obj = REGISTRY[name]()
    if opponent:
        obj.set_opponent(opponent)
    return obj

def create_trainer(name=None):
    if not name:
        name = g_args("trainer")
    if name not in REGISTRY:
        return
    return REGISTRY[name]()