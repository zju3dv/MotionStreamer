import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric, recover_from_root_rot6d)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2
from .humanml.common.skeleton import Skeleton
import torch.nn.functional as F


class HumanML3D_272_DataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        
        self.save_hyperparameters(logger=False)
        self.name = "humanml3d_272"
        self.njoints = 22
        self.hparams['njoints']=22
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            if cfg.TRAIN.STAGE in ['gpt'] and (not cfg.TEST.inference_vq_code):
                if cfg.model.vae_type in ['humanvq']:
                    self.Dataset = Text2MotionDatasetV2_VQToken
                elif cfg.model.vae_type in ['hvq']:
                    self.Dataset = Text2MotionDatasetV2_Dual_codebook_VQToken
                else:
                    raise NotImplementedError
            elif cfg.TEST.inference_vq_code:
                self.Dataset = VQMotionDataset
            else:
                self.Dataset = Text2MotionDatasetV2
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }

        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        
        self.nfeats = self._sample_set.nfeats

    def recover_from_local_position(self, final_x, njoint):
        
        def accumulate_rotations(relative_rotations):
            R_total = [relative_rotations[0]]
            for R_rel in relative_rotations[1:]:
                R_total.append(np.matmul(R_rel, R_total[-1]))
            
            return np.array(R_total)
        
        def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
            a1, a2 = d6[..., :3], d6[..., 3:]
            b1 = F.normalize(a1, dim=-1)
            b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
            b2 = F.normalize(b2, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)
            return torch.stack((b1, b2, b3), dim=-2)
        
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,8:8+3*njoint].reshape(nfrm, -1, 3)
        velocities_root_xy_no_heading = final_x[:,:2] 
        global_heading_diff_rot = final_x[:,2:8] 

        global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), positions_no_heading[...,None]).squeeze(-1)
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
        positions_with_heading[:, :, 0] += root_translation[:, 0:1]
        positions_with_heading[:, :, 2] += root_translation[:, 2:]

        return positions_with_heading

    def feats2joints(self, features, skel=None, motion_type=''):
        assert motion_type in ['']
        assert features.shape[2] == 272
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return self.recover_from_local_position(features.reshape(-1, 272).detach().cpu().numpy(), self.njoints).reshape(features.shape[0], -1, 22, 3)
        

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        return features

    def renorm4t2m(self, features):
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def renorm2ori(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean

        return features


    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
