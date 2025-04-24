import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

from mld.data.humanml.scripts.motion_process import (qrot,
                                                     recover_root_rot_pos)

from .infonce import InfoNCE


class GPTLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        # self.vae_type = cfg.model.vae_type
        # self.mode = mode
        self.cfg = cfg
        # self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE

        assert self.stage in ["gpt"]
        losses = []

        # diffusion loss
        # if self.stage in ['diffusion', 'vae_diffusion']:
        #     # instance noise loss
        #     losses.append("inst_loss")
        #     losses.append("x_loss")
        #     if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
        #         # prior noise loss
        #         losses.append("prior_loss")

        # if self.stage in ['vae', 'vae_diffusion']:
        #     # reconstruction loss
        #     losses.append("recons_feature")
        #     losses.append("recons_verts")
        #     losses.append("recons_joints")
        #     losses.append("recons_limb")

        #     losses.append("gen_feature")
        #     losses.append("gen_joints")

        #     # KL loss
        #     if self.vae_type in ['mld_dual_vae']:
        #         losses.append("kl_motionbody")
        #         losses.append("kl_motionhand")
        #     else:
        #         losses.append("kl_motion")
            
        #     # import pdb; pdb.set_trace()
        #     # vel Loss
        #     if cfg.LOSS.Velocity_loss:
        #         losses.append("recons_velocity")

        # if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:
        #     raise ValueError(f"Stage {self.stage} not supported")

        losses.append("ce_motiontoken")
        if self.cfg.TRAIN.use_tmr_supervision:
            losses.append("contrastive_tmrsupervise")
            self.infonce_temp = cfg.LOSS.INFONCE_TEMP
            
            # self.add_state("count", torch.tensor(0), dist_reduce_fx="mean")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

        if self.stage in ['gpt']:
            self.add_state("rightnum",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.add_state("count_all_token",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")

        self.losses = losses


        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            elif loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            elif loss.split('_')[0] == 'ce':
                self._losses_func[loss] = torch.nn.CrossEntropyLoss(
                    reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'contrastive':
                self._losses_func[loss] = InfoNCE(self.infonce_temp)
                self._params[loss] = cfg.LOSS.LAMBDA_INFONCE
            else:
                ValueError("This loss is not recognized.")

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        # import pdb; pdb.set_trace()
        assert len(rs_set['m_rst']) == len(rs_set['m_ref'])
        bs = len(rs_set['m_rst'])
        
        if self.stage in ['gpt']:

            if self.cfg.TRAIN.use_tmr_supervision:
                total += self._update_loss("contrastive_tmrsupervise", (rs_set['supervise_motion_feat'], rs_set['supervise_text_feat']), rs_set['emb_dist'])

            for i in range(bs):
                total += self._update_loss("ce_motiontoken", rs_set['m_rst'][i], rs_set['m_ref'][i]) / bs # rs_set['m_rst'][i] (16, 513) rs_set['m_ref'][i] (16)
                # import pdb; pdb.set_trace()
                probs = torch.softmax(rs_set['m_rst'][i], dim=-1)
                _, cls_pred_index = torch.max(probs, dim=-1) # 16
                self.count_all_token += cls_pred_index.shape[0]
                self.rightnum += (cls_pred_index.flatten(0) == rs_set['m_ref'][i].flatten(0)).sum().item()

        # import pdb; pdb.set_trace()
        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        loss_dict = {loss: getattr(self, loss) / count for loss in self.losses}
        # import pdb; pdb.set_trace()
        loss_dict['ACC_token'] = self.rightnum / self.count_all_token
        return loss_dict

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name


class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
