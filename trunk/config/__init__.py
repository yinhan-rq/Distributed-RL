import yaml

g_global_config = None

def global_args(param):
    global g_global_config
    if not g_global_config:
        return
    return g_global_config.get(param)


def init_global_args(name,scene=""):
    global g_global_config
    if(scene):
        name = name + "_" +scene
    with open(f"./config/{name}.yaml", "r",encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    g_global_config = config_dict