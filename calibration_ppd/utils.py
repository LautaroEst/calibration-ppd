from importlib.machinery import SourceFileLoader
from types import ModuleType

ROOT_DIR = "/".join(__file__.split("/")[:-2])
DATA_DIR = ROOT_DIR + "/data"
DATASETS_DIR = DATA_DIR + "/datasets"
MODELS_DIR = DATA_DIR + "/models"
SCRIPTS_DIR = ROOT_DIR + "/scripts"
PROJECTS_DIR = ROOT_DIR + "/projects"


def import_configs_objs(config_file):
    """Dynamicaly loads the configuration file"""
    if config_file is None:
        raise ValueError("No config path")
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod