import importlib


def get_model(cfg, datamodule, phase="train"):
    modeltype = cfg.model.model_type
    if modeltype in ["mld", "temos", "gpt"]:
        # import pdb; pdb.set_trace()
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    # import pdb; pdb.set_trace()
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="mld.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    # import pdb; pdb.set_trace()
    return Model(cfg=cfg, datamodule=datamodule)
