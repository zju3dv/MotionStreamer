import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os


def get_module_config(cfg_model, path="modules"):
    module_conf = OmegaConf.create()
    files = os.listdir(f'./configs/{path}/')
    for file in files:
        if file.endswith('.yaml'):
            with open(f'./configs/{path}/' + file, 'r') as f:
                module_conf.merge_with(OmegaConf.load(f))
    module_conf.merge_with(cfg_model)
    return module_conf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):
    parser = ArgumentParser()

    group = parser.add_argument_group("Training options")
    if phase in ["train", "test"]:
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/config.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           help="evaluate existing npys")

    # remove None params, and create a dictionnary
    params = parser.parse_args()
    # params = {key: val for key, val in vars(opt).items() if val is not None}

    # update config from files
    cfg_base = OmegaConf.load('./configs/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_exp.model = cfg_model
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = (params.batch_size
                                if params.batch_size else cfg.TRAIN.BATCH_SIZE)
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        cfg.DEBUG = False if phase == "test" else cfg.DEBUG
        if phase == "test":
            cfg.DEBUG = False
            cfg.DEVICE = [0]
            print("Force no debugging and one gpu when testing")
        cfg.TEST.TEST_DIR = params.dir if params.dir else cfg.TEST.TEST_DIR

    # debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.OFFLINE = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    return cfg
