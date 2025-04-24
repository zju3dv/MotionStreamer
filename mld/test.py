import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from collections import OrderedDict

use_differnt_t2m=False
def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")



def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))
    # import pdb; pdb.set_trace()
    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.model_type))
    

    # optimizer
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
    ]
    logger.info("Callbacks initialized")

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    

    # import pdb; pdb.set_trace()

    if cfg.EVAL.use_tmr_eval:
        t2m_checkpoint = t2m_checkpoint = torch.load(('/comp_robot/lushunlin/motion-latent-diffusion/deps/t2m/motionx/version1/ric_rot/root_body_pos_vel_hand_pos_vel/TMR_pretrain/epoch=699.ckpt'),  map_location=torch.device('cpu'))
        tmr_state_dict = t2m_checkpoint["state_dict"]

        tmr_textencoder_dict = OrderedDict()
        for k, v in tmr_state_dict.items():
            # print(k)
            if k.split(".")[0] == "textencoder":
                name = k.replace("textencoder", "t2m_TMR_textencoder_eval")
                tmr_textencoder_dict[name] = v

        tmr_motionencoder_dict = OrderedDict()
        for k, v in tmr_state_dict.items():
            # print(k)
            if k.split(".")[0] == "motionencoder":
                name = k.replace("motionencoder", "t2m_TMR_motionencoder_eval")
                tmr_motionencoder_dict[name] = v
            
        # import pdb; pdb.set_trace()
        state_dict.update(tmr_textencoder_dict)
        state_dict.update(tmr_motionencoder_dict)

        model.t2m_TMR_textencoder_eval.freeze()
        model.t2m_TMR_motionencoder_eval.freeze()

        model.t2m_TMR_textencoder_eval.eval()
        model.t2m_TMR_motionencoder_eval.eval()
        for p in model.t2m_TMR_textencoder_eval.parameters():
            p.requires_grad = False
        for p in model.t2m_TMR_motionencoder_eval.parameters():
            p.requires_grad = False


    model.load_state_dict(state_dict)

    if use_differnt_t2m:
        # t2m_checkpoint = torch.load('/comp_robot/lushunlin/motion-latent-diffusion/deps/t2m/motionx/version_only_humanml/smplx_212/text_mot_match_glove_6B_caption_bs_256/model/finest.tar')
        t2m_checkpoint = torch.load('/comp_robot/lushunlin/motion-latent-diffusion/deps/t2m/motionx/version1/smplx_212/text_mot_match_glove_6B_caption_bs_256/model/finest.tar')
        model.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        model.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        model.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])
        
        # freeze params
        model.t2m_textencoder.eval()
        model.t2m_moveencoder.eval()
        model.t2m_motionencoder.eval()
        for p in model.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in model.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in model.t2m_motionencoder.parameters():
            p.requires_grad = False

        print('Loading /comp_robot/lushunlin/motion-latent-diffusion/deps/t2m/motionx/version1/smplx_212/text_mot_match_glove_6B_caption_bs_256/model/finest.tar')



    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    # calculate metrics
    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]
        if "TM2TMetrics" in metrics_type:
            # mm meteics
            logger.info(f"Evaluating MultiModality - Replication {i}")
            # import pdb; pdb.set_trace()
            datasets.mm_mode(True)
            mm_metrics = trainer.test(model, datamodule=datasets)[0]
            metrics.update(mm_metrics)
            datasets.mm_mode(False)
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    # metrics = trainer.validate(model, datamodule=datasets[0])
    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
