import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric, recover_from_body_pos_vel_hand_rot)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetMotionX, Text2MotionDatasetMotionX_text_all, VQMotionDataset_MotionX, Text2MotionDatasetMotionX_dual_codebook, Text2MotionDatasetMotionX_VQToken


class Motion_X_V25_DataModule(BASEDataModule):

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
        # import pdb; pdb.set_trace()
        assert len(eval(f'cfg.{phase.upper()}.DATASETS')) == 1
        assert eval(f'cfg.{phase.upper()}.DATASETS')[0] == 'motionx_v25'

        self.name = "motionx_v25"

        if cfg.DATASET.JOINT_TYPE == 'humanml3d':
            self.njoints = 22
            self.hparams['njoints']=22
        elif cfg.DATASET.JOINT_TYPE in ['motionx', 'motionx_v25']:

            if 'MINOR_MOTION_TYPE' in cfg.DATASET:
                if cfg.DATASET.MINOR_MOTION_TYPE in ['root_position_vel_only_body']:
                    self.njoints = 22
                    self.hparams['njoints']=22
                else:
                    self.njoints = 52
                    self.hparams['njoints']=52

            else:
                self.njoints = 52
                self.hparams['njoints']=52

        else:
             raise NotImplentError

        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            if cfg.TRAIN.STAGE in ['gpt'] and (not cfg.TEST.inference_vq_code):
                if cfg.model.vae_type in ['humanvq']:
                    # import pdb; pdb.set_trace()
                    self.Dataset = Text2MotionDatasetMotionX_VQToken
                elif cfg.model.vae_type in ['hvq', 'hvq_body_hand']:
                    self.Dataset = Text2MotionDatasetMotionX_dual_codebook
                else:
                    raise NotImplentmentError
            elif cfg.TEST.inference_vq_code:
                self.Dataset = VQMotionDataset_MotionX
            else:
                if cfg.model.condition in ['text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', "text_seperate", "only_pose_concat", "only_pose_fusion"]:
                    self.Dataset = Text2MotionDatasetMotionX_text_all
                else:
                    self.Dataset = Text2MotionDatasetMotionX
                
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # import pdb; pdb.set_trace()
        # self.transforms = self._sample_set.transforms

    def recover_from_root_body_pos_vel_hand_pos_vel_hand_wrist(self, motion):
        # import pdb; pdb.set_trace()
        body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
        left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].view(motion.shape[0], motion.shape[1], 15, 3) + torch.unsqueeze(body_pos_motion[..., -6:-3], dim=-2)).reshape(motion.shape[0], motion.shape[1], -1) # 45
        right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].view(motion.shape[0], motion.shape[1], 15, 3) + torch.unsqueeze(body_pos_motion[..., -3:], dim=-2)).reshape(motion.shape[0], motion.shape[1], -1) # 45

        body_vel_motion = motion[..., 4+(52 - 1) * 3: 4+(52 - 1) * 3 + 22*3] # 66
        left_hand_vel_motion = (motion[..., 4+(52 - 1) * 3 + 22*3: 4+(52 - 1) * 3 + 22*3 + 15 * 3].view(motion.shape[0], motion.shape[1], 15, 3) + torch.unsqueeze(body_vel_motion[..., -6:-3], dim=-2)).reshape(motion.shape[0], motion.shape[1], -1)
        right_hand_vel_motion = (motion[..., 4+(52 - 1) * 3 + 22*3 + 15 * 3: ].view(motion.shape[0], motion.shape[1], 15, 3) + torch.unsqueeze(body_vel_motion[..., -3:], dim=-2)).reshape(motion.shape[0], motion.shape[1], -1)

        motion = torch.cat((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)

        return motion

    def feats2joints(self, features, motion_type, skel=None, smplx_model=None):
        # import pdb; pdb.set_trace()
        if motion_type in ['all', 'root_body_pos_vel_hand_all', 'root_body_pos_vel_hand_pos_vel', 'root_body_pos_vel_hand_pos']:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            return recover_from_ric(features, self.njoints) # torch.Size([32, 92, 22, 3])
        elif motion_type in ['root_body_pos_vel_hand_pos_vel_hand_wrist']:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            features = self.recover_from_root_body_pos_vel_hand_pos_vel_hand_wrist(features)
            return recover_from_ric(features, self.njoints) # torch.Size([32, 92, 22, 3])
        elif motion_type in ['root_position_vel_only_body']:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            return recover_from_ric(features, 22)
        elif motion_type in ['smplx_212']:
            assert smplx_model is not None
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * (std + 1e-7) + mean
            bs = features.shape[0]
            features = features.reshape(-1, 212)
            output = smplx_model.smplx_model(pose_body=features[:,3:66], pose_hand=features[:,66:156], root_orient=features[:,:3]).Jtr
            return output.reshape(bs, -1, 55, 3) #torch.Size([32, 96, 55, 3])
        elif motion_type in ['root_body_pos_vel_hand_rot']:
            assert len(skel) == 2

            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            return recover_from_body_pos_vel_hand_rot(features, self.njoints, skel)

        else:
            raise NotImplementedError
            
            

    def joints2feats(self, features):
        import pdb; pdb,set_trace()
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # import pdb; pdb.set_trace()
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * (ori_std + 1e-7) + ori_mean
        features = (features - eval_mean) / (eval_std + 1e-7)
        return features

    def renorm2ori(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * (std + 1e-7)  + mean

        return features
    
    def renormt2m_back(self, features):
        import pdb; pdb.set_trace()
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * (eval_std + 1e-7) + eval_mean
        return features


    def mm_mode(self, mm_on=True):
        # random select samples for mm
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
