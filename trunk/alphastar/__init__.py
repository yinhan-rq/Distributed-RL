from .actor import ActorLoop
from .selfplay import League
from config import global_args as g_args

def start(base_model=None):
    league = League(base_model, g_args("main_agents"), g_args("main_exploiters"), g_args("league_exploiters"))
    actors = []
    for player in league.get_all_players():
        actors.append(ActorLoop(player, league))
    for a in actors:
        a.start()