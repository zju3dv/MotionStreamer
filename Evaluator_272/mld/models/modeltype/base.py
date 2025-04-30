import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
# from mld.models.metrics import ComputeMetrics, MRMetrics, TM2TMetrics, TM2TMetrics_R256, MMMetrics, HUMANACTMetrics, UESTCMetrics, UncondMetrics, ComputeMetrics_body_hand, MRMetrics_body_hand, ACCMetrics, TMR_TM2TMetrics
from mld.models.metrics import TMR_TM2TMetrics
from os.path import join as pjoin
from collections import OrderedDict


class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val", "test"]:

            if self.trainer.datamodule.is_mm and ("TM2TMetrics" in self.metrics_dict or "TM2TMetrics_R256" in self.metrics_dict):
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(
                    self,
                    metric).compute(sanity_flag=self.trainer.sanity_checking)
                # reset metrics
                getattr(self, metric).reset()
                dico.update({
                    f"Metrics/{metric}": value.item()
                    for metric, value in metrics_dict.items()
                })
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        # # ToDo
        # # re-write vislization checkpoint?
        # # visualize validation
        # parameters = {"xx",xx}
        # vis_path = viz_epoch(self, dataset, epoch, parameters, module=None,
        #                         folder=parameters["folder"], writer=None, exps=f"_{dataset_val.dataset_name}_"+val_set)
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self.save_npy(outputs)
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)

    def on_save_checkpoint(self, checkpoint):
        # don't save clip to checkpoint
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # restore clip state_dict to checkpoint
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in checkpoint['state_dict'].items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # load clip state_dict to checkpoint
        if hasattr(self, 'text_encoder'):
            clip_state_dict = self.text_encoder.state_dict()
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                new_state_dict['text_encoder.' + k] = v
            for k, v in state_dict.items():
                if 'text_encoder' not in k:
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict

        super().load_state_dict(new_state_dict, strict)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == "TemosMetric":
                self.TemosMetric = ComputeMetrics(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )

            elif metric == "TemosMetric_body_hand":
                self.TemosMetric_body_hand = ComputeMetrics_body_hand(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )

            elif metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == 'TM2TMetrics_R256':
                self.TM2TMetrics_R256 = TM2TMetrics_R256(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == 'TMR_TM2TMetrics':
                self.TMR_TM2TMetrics = TMR_TM2TMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "MRMetrics":
                self.MRMetrics = MRMetrics(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )

            elif metric == "MRMetrics_body_hand":
                self.MRMetrics_body_hand = MRMetrics_body_hand(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )

            elif metric == "HUMANACTMetrics":
                self.HUMANACTMetrics = HUMANACTMetrics(
                    datapath=os.path.join(self.cfg.model.humanact12_rec_path,
                                          "humanact12_gru.tar"),
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UESTCMetrics":
                self.UESTCMetrics = UESTCMetrics(
                    cfg=self.cfg,
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    multimodality_times=self.cfg.TEST.MM_NUM_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "UncondMetrics":
                self.UncondMetrics = UncondMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "ACCMetrics":
                self.ACCMetrics = ACCMetrics(dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
            else:
                raise NotImplementedError(
                    f"Do not support Metric Type {metric}")
        if "TM2TMetrics" in self.metrics_dict or "UncondMetrics" in self.metrics_dict or "TM2TMetrics_R256" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
            )

    def save_npy(self, outputs):
        
        cfg = self.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.model_type),
                str(cfg.NAME),
                "samples",
            ))

        if cfg.TEST.SAVE_PREDICTIONS and cfg.TEST.REP_I + 1 == cfg.TEST.REPLICATION_TIMES:
            if cfg.TEST.inference_vq_code:
                if self.vae_type in ["hvq", "hvq_body_hand"]:
                    name = [i[2] for i in outputs]
                    motion_code_t = [i[0] for i in outputs]
                    motion_code_b = [i[1] for i in outputs]
                else:
                    name = [i[1] for i in outputs]
                    outputs = [i[0] for i in outputs]

            else:
                if cfg.DATASET.MOTION_TYPE == 'vector_263':
                    lengths = [i[1] for i in outputs]
                    texts = [i[2] for i in outputs]
                    outputs = [i[0] for i in outputs]
                elif cfg.DATASET.MOTION_TYPE == 'smplx_212':
                    if cfg.TRAIN.use_joints:
                        lengths = [i[1] for i in outputs]
                        gen_motions = [self.datamodule.renormt2m_back(i[0]) for i in outputs]
                        ref_motions = [self.datamodule.renormt2m_back(i[2]) for i in outputs]
                    else:
                        return
                elif cfg.DATASET.MOTION_TYPE in ['ric_rot']:
                    lengths = [i[1] for i in outputs]
                    gen_motions = [i[0] for i in outputs]
                    ref_motions = [i[2] for i in outputs]
                else:
                    raise NotImplementedError
           
            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                if cfg.TEST.inference_vq_code:
                    for i in range(len(outputs)):
                        if self.vae_type in ["hvq", "hvq_body_hand"]:
                            for bid in range(
                                    min(cfg.TEST.BATCH_SIZE, motion_code_t[i].shape[0])):
                                
                                motion_vqcode_t = motion_code_t[i][bid].cpu().numpy()[None, :]
                                motion_vqcode_b = motion_code_b[i][bid].cpu().numpy()[None, :]
                                motion_name = name[i][bid]
                                
                                assert cfg.TEST.REPLICATION_TIMES == 1

                                motion_name = f"{motion_name}.npy"
                                output_dir_t = Path(
                                    os.path.join(f'./datasets/{cfg.TEST.DATASETS[0]}/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_t'))
                                output_dir_b = Path(
                                    os.path.join(f'./datasets/{cfg.TEST.DATASETS[0]}/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_b'))
                                # save predictions results
                                npypath_t = output_dir_t / motion_name
                                npypath_b = output_dir_b / motion_name

                                np.save(npypath_t, motion_vqcode_t)
                                np.save(npypath_b, motion_vqcode_b)



                        else:
                            for bid in range(
                                    min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                                motion_vqcode = outputs[i][bid].cpu().numpy()[None, :]
                                motion_name = name[i][bid]
                                
                                assert cfg.TEST.REPLICATION_TIMES == 1

                                motion_name = f"{motion_name}.npy"
                                output_dir = Path(
                                    os.path.join(f'./datasets/{cfg.TEST.DATASETS[0]}/vq_tokens', str(cfg.model.vae_type)))
                                # save predictions results
                                npypath = output_dir / motion_name
                                np.save(npypath, motion_vqcode)


                else:
                    keyids = self.trainer.datamodule.test_dataset.name_list
                    for i in range(len(outputs)):
                        for bid in range(
                                min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                            keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                            gen_joints = outputs[i][bid].cpu().numpy()
                            text = texts[i][bid]
                            
                            if cfg.TEST.REPLICATION_TIMES > 1:
                                name = f"{keyid}_{cfg.TEST.REP_I}"
                            else:
                                name = f"{keyid}.npy"
                            # save predictions results
                            npypath = output_dir / name
                            np.save(npypath, gen_joints)

                            textpath = output_dir / 'text' / (name + '.txt')
                            os.makedirs(os.path.split(textpath)[0], exist_ok=True)
                            with open(textpath, "w") as f:
                                f.write(text)
            elif cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
                keyids = range(len(self.trainer.datamodule.test_dataset))
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu()
                        gen_joints = gen_joints.permute(2, 0,
                                                        1)[:lengths[i][bid],
                                                           ...].numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
            elif cfg.TEST.DATASETS[0].lower() in ["motionx", 'motionx_v26']:


                if cfg.TEST.inference_vq_code:
                    for i in range(len(outputs)):
                        if self.vae_type in ["hvq", "hvq_body_hand"]:
                            for bid in range(
                                    min(cfg.TEST.BATCH_SIZE, motion_code_t[i].shape[0])):
                                motion_vqcode_t = motion_code_t[i][bid].cpu().numpy()[None, :]
                                motion_vqcode_b = motion_code_b[i][bid].cpu().numpy()[None, :]
                                motion_name = name[i][bid]
                                
                                assert cfg.TEST.REPLICATION_TIMES == 1

                                motion_name = f"{motion_name}.npy"
                                if cfg.TEST.DATASETS[0].lower() == 'motionx_v26':
                                    output_dir_t = Path(
                                        os.path.join(f'./datasets/Motion-X-V26/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_t'))
                                    output_dir_b = Path(
                                        os.path.join(f'./datasets/Motion-X-V26/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_b'))
                                elif cfg.TEST.DATASETS[0].lower() == 'motionx':
                                    output_dir_t = Path(
                                        os.path.join(f'./datasets/Motion-X/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_t'))
                                    output_dir_b = Path(
                                        os.path.join(f'./datasets/Motion-X/vq_tokens', str(cfg.model.vae_type), 'motion_vqcode_b'))
                                else:
                                    raise NotImplementedError
                                # save predictions results

                                npypath_t = output_dir_t / motion_name
                                npypath_b = output_dir_b / motion_name

                                npypath_t_ref_parent_directory = os.path.dirname(npypath_t)
                                if not os.path.exists(npypath_t_ref_parent_directory):
                                    os.makedirs(npypath_t_ref_parent_directory)

                                npypath_b_parent_directory = os.path.dirname(npypath_b)
                                if not os.path.exists(npypath_b_parent_directory):
                                    os.makedirs(npypath_b_parent_directory)

                                np.save(npypath_t, motion_vqcode_t)
                                np.save(npypath_b, motion_vqcode_b)


                            

                        else:
                            for bid in range(
                                    min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                                motion_vqcode = outputs[i][bid].cpu().numpy()[None, :]
                                motion_name = name[i][bid]
                                
                                assert cfg.TEST.REPLICATION_TIMES == 1

                                motion_name = f"{motion_name}.npy"
                                output_dir = Path(
                                    os.path.join(f'./datasets/Motion-X/vq_tokens', str(cfg.model.vae_type)))
                                # save predictions results

                                npypath = output_dir / motion_name
                                npypath_parent_directory = os.path.dirname(npypath)
                                if not os.path.exists(npypath_parent_directory):
                                    os.makedirs(npypath_parent_directory)
                                np.save(npypath, motion_vqcode)



                else:

                    keyids = self.trainer.datamodule.test_dataset.name_list
                    for i in range(len(gen_motions)):
                        for bid in range(
                                min(cfg.TEST.BATCH_SIZE, gen_motions[i].shape[0])):
                            keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                            gen_joints = gen_motions[i][bid].cpu().numpy()
                            ref_joints = ref_motions[i][bid].cpu().numpy()

                            gen_name = f"{keyid}.npy"
                            ref_name = f"{keyid}_gt.npy"
                            # save predictions results
                            npypath = output_dir / gen_name
                            os.makedirs(os.path.split(npypath)[0], exist_ok=True)
                            np.save(npypath, gen_joints)
                            

