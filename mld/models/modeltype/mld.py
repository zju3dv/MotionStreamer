import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_dual_vae, 
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses, MLDLosses_no_joint
from mld.models.losses.vqvae import VQVAELosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding

from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

from .base import BaseModel
from .smplx_layer import smplx_layer

from ..body_skeleton.skeleton import Skeleton
from ..body_skeleton.paramUtil import *

from collections import OrderedDict
from sentence_transformers import SentenceTransformer

import copy


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        # import pdb; pdb.set_trace()
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        if 'MINOR_MOTION_TYPE' in cfg.DATASET:
            self.input_format = cfg.DATASET.MINOR_MOTION_TYPE
        else:
            self.input_format = cfg.DATASET.MOTION_TYPE

        self.motion_type = cfg.DATASET.MOTION_TYPE

        self.eval_on_text = cfg.EVAL.eval_on_text
        # 
        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        self.smplx_model = smplx_layer()

        self.smplx_model.eval()
        for p in self.smplx_model.parameters():
            p.requires_grad = False

        # import pdb; pdb.set_trace()
        if self.vae_type != "no":
            # 
            self.vae = instantiate_from_config(cfg.model.motion_vae)
        # import pdb; pdb.set_trace()
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert", "actor", "humanvq"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        # import pdb; pdb.set_trace()
        # if self.cfg.TRAIN.STAGE not in ["vae"]:
        if cfg.EVAL.eval_on_text:
            if self.condition in ["text", "text_uncond", 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                self._get_t2m_evaluator(cfg)

        if cfg.EVAL.use_tmr_eval:
            if self.condition in ["text", "text_uncond", 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                self._get_tmr_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            # assert cfg.DATASET.MOTION_TYPE in ['vector_263', 'root_position']
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        elif cfg.LOSS.TYPE == "vqvae":

            self._losses = MetricCollection({
                split: VQVAELosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        elif cfg.LOSS.TYPE == 'mld_no_joint':
            # assert 'smpl' not in cfg.DATASET.MOTION_TYPE
            self._losses = MetricCollection({
                split: MLDLosses_no_joint(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        # if cfg.LOSS.TYPE == 'mld_no_joint':
        #     assert cfg.TRAIN.use_joints == False

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time

        if eval("self.cfg.TRAIN.DATASETS")[0].lower() == 'humanml3d':
            n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
            kinematic_chain = t2m_kinematic_chain
        elif eval("self.cfg.TRAIN.DATASETS")[0].lower() == 'kit':
            n_raw_offsets = torch.from_numpy(kit_raw_offsets)
            kinematic_chain = kit_kinematic_chain
        elif eval("self.cfg.TRAIN.DATASETS")[0].lower() in ['motionx', 'motionx_v25', 'motionx_v26']:
            n_raw_offsets = torch.from_numpy(t2m_raw_body_hand_offsets)
            body_raw_offsets = n_raw_offsets[:22]
            hand_raw_offsets = n_raw_offsets[22:]
            kinematic_chain = t2m_body_hand_kinematic_chain
            body_kinemantic_chain = t2m_kinematic_chain
            hand_kinemantic_chain = t2m_left_hand_chain + t2m_right_hand_chain
        else:
            raise NotImplementedError


        self.skel=None
        if self.input_format in ['root_rot6d']:
            example_data = np.load(os.path.join('/comp_robot/lushunlin/HumanML3D-1/joints', '000021' + '.npy'))
            example_data = example_data.reshape(len(example_data), -1, 3)
            example_data = torch.from_numpy(example_data)
            tgt_skel = Skeleton(n_raw_offsets, kinematic_chain)
            # (joints_num, 3)
            import pdb; pdb.set_trace() #finish select joints
            tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
            self.skel = Skeleton(n_raw_offsets, kinematic_chain)
            self.skel.set_offset(tgt_offsets)
            # import pdb; pdb.set_trace()
        elif self.input_format in ['root_body_pos_vel_hand_rot']:
            # import pdb; pdb.set_trace()
            example_data = np.load('/comp_robot/lushunlin/datasets/Motion-X/motion_data/joint/humanml/000021.npy')
            example_data = example_data.reshape(len(example_data), -1, 3)
            example_data = torch.from_numpy(example_data)

            example_data = example_data[:, :52]

            body_example_data = example_data[:, :22]
            tgt_body_skel = Skeleton(body_raw_offsets, body_kinemantic_chain)

            tgt_skel = Skeleton(n_raw_offsets, kinematic_chain)

            # (joints_num, 3)
            tgt_body_skel_offsets = tgt_body_skel.get_offsets_joints(body_example_data[0])
            tgt_skel_offsets = tgt_skel.get_offsets_joints(example_data[0])

            body_skel = Skeleton(body_raw_offsets, body_kinemantic_chain)
            all_skel = Skeleton(n_raw_offsets, kinematic_chain)

            body_skel.set_offset(tgt_body_skel_offsets)
            all_skel.set_offset(tgt_skel_offsets)

            self.skel = (body_skel, all_skel)
            # self.skel.set_offset(tgt_offsets)

        

        
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond', "text_all", 'text_body', 'text_hand', 'text_face_body', 'text_face', "text_seperate", "only_pose_concat", "only_pose_fusion"]:
            self.feats2joints = datamodule.feats2joints
            self.renorm2ori = datamodule.renorm2ori
            if self.cfg.model.vae_type == 'hvq_body_hand_face':
                self.facerenorm2ori = datamodule.facerenorm2ori
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """

        
        # init module
        if cfg.model.eval_text_source == 'token':

            self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(word_size=cfg.model.t2m_textencoder.dim_word,
                                        pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
                                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                       )
        elif cfg.model.eval_text_source == 'only_text_token':

            self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(word_size=cfg.model.t2m_textencoder.dim_word,
                                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                       )

        elif cfg.model.eval_text_source in ['caption']:


            if cfg.model.eval_text_encode_way == 'clip':
                self.t2m_textencoder, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
                clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
                self.t2m_textencoder.eval()
                for p in text_enc.parameters():
                    p.requires_grad = False

            elif cfg.model.eval_text_encode_way == 't5':
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.t2m_textencoder = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
                self.t2m_textencoder.eval()
                for p in self.t2m_textencoder.parameters():
                    p.requires_grad = False

            elif 'GRU' in cfg.model.eval_text_encode_way:
                self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(word_size=cfg.model.t2m_textencoder.dim_word,
                                            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                        )
            else:
                raise NotImplementedError

        
        # import pdb; pdb.set_trace()
        if cfg.DATASET.MOTION_TYPE in ['vector_263', 'ric_rot', 'vector_263_ori_humanml']:
            self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
                input_size=cfg.DATASET.NFEATS - 4,
                hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
                output_size=cfg.model.t2m_motionencoder.dim_move_latent,
            )
        elif cfg.DATASET.MOTION_TYPE in ['smplx_212', 'smplx_159']:
            self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
                input_size=cfg.DATASET.NFEATS,
                hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
                output_size=cfg.model.t2m_motionencoder.dim_move_latent,
            )
        
        else:
            raise NotImplementedError

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )

        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        # t2m_checkpoint = torch.load(
        #     os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, 
        #                  "text_mot_match_glove_6B_caption_bs_256/model/finest.tar"))

        # import pdb; pdb.set_trace()
        minor_motin_type = cfg.DATASET.MINOR_MOTION_TYPE if 'MINOR_MOTION_TYPE' in cfg.DATASET else ''
        # import pdb; pdb.set_trace()
        if dataname in ['motionx', 'motionx_v25', 'motionx_v26']:
            # import pdb; pdb.set_trace()
            if 'TEXT_TYPE' in cfg.DATASET:
                if cfg.DATASET.TEXT_TYPE == 'vicuna1.5_13b':
                    # import pdb; pdb.set_trace()
                    t2m_checkpoint = torch.load(
                        os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, minor_motin_type, 
                                    "text_mot_match_glove_6B_caption_bs_256_text_vicuna1.5/model/finest.tar"),  map_location=torch.device('cpu'))
                elif cfg.DATASET.TEXT_TYPE == 'vicuna1.5_13b_add_subject':
                    t2m_checkpoint = torch.load(
                        os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, minor_motin_type, 
                                    "text_mot_match_glove_6B_caption_bs_256_text_vicuna1.5_add_subject/model/finest.tar"),  map_location=torch.device('cpu'))

            else:
                t2m_checkpoint = torch.load(
                    os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, minor_motin_type, 
                                "text_mot_match_glove_6B_caption_bs_256/model/finest.tar"),  map_location=torch.device('cpu'))
        else:
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname,
                            "text_mot_match/model/finest.tar"),  map_location=torch.device('cpu'))

        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])


        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False


    def _get_tmr_t2m_evaluator(self, cfg):
        """
        load tmr T2M text encoder and motion encoder for evaluating
        """

        # import pdb; pdb.set_trace()
        # init module

        assert cfg.model.eval_text_source in ['caption']

        self.t2m_TMR_textencoder_eval = DistilbertActorAgnosticEncoder('distilbert-base-uncased', num_layers=4)
        self.t2m_TMR_motionencoder_eval = ActorAgnosticEncoder(nfeats=cfg.DATASET.NFEATS, vae =True, num_layers=4)
        

        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        # t2m_checkpoint = torch.load(
        #     os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, 
        #                  "text_mot_match_glove_6B_caption_bs_256/model/finest.tar"))


        minor_motin_type = cfg.DATASET.MINOR_MOTION_TYPE if 'MINOR_MOTION_TYPE' in cfg.DATASET else ''
        # import pdb; pdb.set_trace()
        if dataname in ['motionx', 'motionx_v25', 'motionx_v26']:
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, minor_motin_type, "TMR_pretrain_new/epoch=59.ckpt"),  map_location=torch.device('cpu'))
            state_dict = t2m_checkpoint["state_dict"]
        else:
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname,
                            "text_mot_match/model/finest.tar"),  map_location=torch.device('cpu'))
            
        tmr_textencoder_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if k.split(".")[0] == "textencoder":
                name = k.replace("textencoder.", "")
                tmr_textencoder_dict[name] = v

        self.t2m_TMR_textencoder_eval.load_state_dict(tmr_textencoder_dict, strict=True)
        

        tmr_motionencoder_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if k.split(".")[0] == "motionencoder":
                name = k.replace("motionencoder.", "")
                tmr_motionencoder_dict[name] = v
        
        self.t2m_TMR_motionencoder_eval.load_state_dict(tmr_motionencoder_dict, strict=True)

        # import pdb; pdb.set_trace()
        # freeze params
        self.t2m_TMR_textencoder_eval.freeze()
        self.t2m_TMR_motionencoder_eval.freeze()
        self.t2m_TMR_textencoder_eval.eval()
        self.t2m_TMR_motionencoder_eval.eval()
        for p in self.t2m_TMR_textencoder_eval.parameters():
            p.requires_grad = False
        for p in self.t2m_TMR_motionencoder_eval.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        import pdb; pdb.set_trace()
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set





    def train_vae_forward(self, batch):
        # import pdb; pdb.set_trace()
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["hvq_body_hand_face"]:
            face_ref = batch["face_motion"]
            # import pdb; pdb.set_trace()


        joint_mask = batch["joint_mask"]
        # import pdb; pdb.set_trace()
        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
            feats_rst, (commit_x, commit_x_d), perplexity = self.vae.forward(feats_ref)
        elif self.vae_type in ["hvq"]:
            feats_rst, (commit_x_t, commit_x_d_t), (commit_x_b, commit_x_d_b) = self.vae.forward(feats_ref)
            # import pdb; pdb.set_trace()
        elif self.vae_type in ["hvq_body_hand"]:
            feats_rst, (commit_x_t, commit_x_d_t), (commit_x_b, commit_x_d_b) = self.vae.forward(feats_ref)
        elif self.vae_type in ["hvq_body_hand_face"]:
            feats_rst, (commit_x_f, commit_x_d_f), (commit_x_t, commit_x_d_t), (commit_x_b, commit_x_d_b) = self.vae.forward(torch.cat((feats_ref, face_ref), dim=2))
            face_rst = feats_rst[:, :, -53:]
            feats_rst = feats_rst[:, :, :-53]
            # import pdb; pdb.set_trace()
        elif self.vae_type in ["rvq"]:
            feats_rst, (commit_x, commit_x_d1, commit_x_d2), perplexity = self.vae.forward(feats_ref)

        elif self.vae_type in ["mld_dual_vae"]:
            body_motion_z, hand_motion_z, body_dist_m, hand_dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(body_motion_z, hand_motion_z, lengths)
        elif self.vae_type in ["dual_human_vq"]:
            feats_rst, (body_commit_x, body_commit_x_d), (hand_commit_x, hand_commit_x_d), body_perplexity, hand_perplexity = self.vae.forward(feats_ref)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        if self.vae_type in ["mld", "vposert", "actor"]:
            recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
        elif self.vae_type in ["mld_dual_vae"]:
            body_recons_z, hand_recons_z, body_dist_rm, hand_dist_rm = self.vae.encode(feats_ref, lengths)

        # import pdb; pdb.set_trace()
        # joints recover
        if self.condition in ["text", "text_all", 'text_hand', 'text_body', 'text_face', "text_seperate", "only_pose_concat", "only_pose_fusion"]:

            if self.input_format in ['vector_263', 'vector_263_ori_humanml', 'root_position', 'root_position_vel', 'root_position_rot6d', 'all', 'root_body_pos_vel_hand_all', 'root_body_pos_vel_hand_pos_vel', 'root_body_pos_vel_hand_pos', 'root_position_vel_only_body', 'root_body_pos_vel_hand_pos_vel_hand_wrist']:
                joints_rst = self.feats2joints(feats_rst, self.input_format) # feats_rst.shape (bs, seq, 67) joints_rst.shape (bs, seq, 22, 3)
                joints_ref = self.feats2joints(feats_ref, self.input_format)
            elif self.input_format in ['root_rot6d']:
                joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.input_format)
                joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
                joints_ref = self.feats2joints(feats_ref, skel=self.skel, motion_type=self.input_format)
                joints_ref = joints_ref.view(feats_ref.shape[0], feats_ref.shape[1], self.njoints, 3)
                # import pdb; pdb.set_trace()
            elif self.input_format in ['smplx_212', 'smplx_159'] and self.cfg.TRAIN.use_joints:
                joints_rst = self.feats2joints(feats_rst, self.input_format, self.smplx_model)
                joints_ref = self.feats2joints(feats_ref, self.input_format, self.smplx_model)
            elif self.input_format == 'root_body_pos_vel_hand_rot':

                joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.input_format)
                joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
                joints_ref = self.feats2joints(feats_ref, skel=self.skel, motion_type=self.input_format)
                joints_ref = joints_ref.view(feats_ref.shape[0], feats_ref.shape[1], self.njoints, 3)
            elif self.input_format in ['smplx_212', 'smplx_159'] and (not self.cfg.TRAIN.use_joints):
                pass
                
            else:
                raise NotImplementedError

        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if self.vae_type in ["mld", "vposert", "actor"]:
            if dist_m is not None:
                if self.is_vae:
                    # Create a centred normal distribution to compare with
                    mu_ref = torch.zeros_like(dist_m.loc)
                    scale_ref = torch.ones_like(dist_m.scale)
                    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
                else:
                    dist_ref = dist_m

        elif self.vae_type in ["mld_dual_vae"]:
            if body_dist_m is not None:
                if self.is_vae:
                    # Create a centred normal distribution to compare with
                    body_mu_ref = torch.zeros_like(body_dist_m.loc)
                    body_scale_ref = torch.ones_like(body_dist_m.scale)
                    body_dist_ref = torch.distributions.Normal(body_mu_ref, body_scale_ref)
                else:
                    body_dist_ref = body_dist_m

            if hand_dist_m is not None:
                if self.is_vae:
                    # Create a centred normal distribution to compare with
                    hand_mu_ref = torch.zeros_like(hand_dist_m.loc)
                    hand_scale_ref = torch.ones_like(hand_dist_m.scale)
                    hand_dist_ref = torch.distributions.Normal(hand_mu_ref, hand_scale_ref)
                else:
                    hand_dist_ref = hand_dist_m

        # import pdb; pdb.set_trace()
        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        # import pdb; pdb.set_trace()
        if self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:

            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "commit_x": commit_x,
                "commit_x_d": commit_x_d
            }

            return rs_set

        elif self.vae_type in ["rvq"]:
            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "commit_x": commit_x,
                "commit_x_d1": commit_x_d1, 
                "commit_x_d2": commit_x_d2 
            }

            return rs_set

        
        elif self.vae_type in ["dual_human_vq"]:
            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "body_commit_x": body_commit_x,
                "hand_commit_x": hand_commit_x,
                "body_commit_x_d": body_commit_x_d, 
                "hand_commit_x_d": hand_commit_x_d, 
            }


            return rs_set


        elif self.vae_type in ["hvq", "hvq_body_hand"]:
            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "commit_x_t": commit_x_t,
                "commit_x_d_t": commit_x_d_t, 
                "commit_x_b": commit_x_b , 
                "commit_x_d_b": commit_x_d_b,
                # ""
            }

        elif self.vae_type in ['hvq_body_hand_face']:
            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "fm_ref": face_ref[:, :min_len, :],
                "fm_rst": face_rst[:, :min_len, :],
                "commit_x_t": commit_x_t,
                "commit_x_d_t": commit_x_d_t, 
                "commit_x_b": commit_x_b , 
                "commit_x_d_b": commit_x_d_b,
                "commit_x_f": commit_x_f,
                "commit_x_d_f": commit_x_d_f
                # ""
            }

            # return rs_set

        


        # import pdb; pdb.set_trace()
        if self.vae_type in ['mld_dual_vae']:

            if self.cfg.TRAIN.use_joints:
                rs_set = {
                    "m_ref": feats_ref[:, :min_len, :],
                    "m_rst": feats_rst[:, :min_len, :],
                    # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                    "body_lat_m": body_motion_z.permute(1, 0, 2),
                    "hand_lat_m": hand_motion_z.permute(1, 0, 2),
                    "body_lat_rm": body_recons_z.permute(1, 0, 2),
                    "hand_lat_rm": hand_recons_z.permute(1, 0, 2),
                    "joints_ref": joints_ref,
                    "joints_rst": joints_rst,
                    "body_dist_m": body_dist_m,
                    "hand_dist_m": hand_dist_m,
                    "body_dist_ref": body_dist_ref,
                    "hand_dist_ref": hand_dist_ref,
                }
            else:

                rs_set = {
                    "m_ref": feats_ref[:, :min_len, :],
                    "m_rst": feats_rst[:, :min_len, :],
                    # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                    "body_lat_m": body_motion_z.permute(1, 0, 2),
                    "hand_lat_m": hand_motion_z.permute(1, 0, 2),
                    "body_lat_rm": body_recons_z.permute(1, 0, 2),
                    "hand_lat_rm": hand_recons_z.permute(1, 0, 2),
                    "body_dist_m": body_dist_m,
                    "hand_dist_m": hand_dist_m,
                    "body_dist_ref": body_dist_ref,
                    "hand_dist_ref": hand_dist_ref,
                }

            # return rs_set

        elif self.vae_type in ["mld"]:
            if self.cfg.TRAIN.use_joints:
                rs_set = {
                    "m_ref": feats_ref[:, :min_len, :],
                    "m_rst": feats_rst[:, :min_len, :],
                    # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                    "lat_m": motion_z.permute(1, 0, 2),
                    "lat_rm": recons_z.permute(1, 0, 2),
                    "joints_ref": joints_ref,
                    "joints_rst": joints_rst,
                    "dist_m": dist_m,
                    "dist_ref": dist_ref,
                }
            else:
                rs_set = {
                    "m_ref": feats_ref[:, :min_len, :],
                    "m_rst": feats_rst[:, :min_len, :],
                    # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                    "lat_m": motion_z.permute(1, 0, 2),
                    "lat_rm": recons_z.permute(1, 0, 2),
                    "dist_m": dist_m,
                    "dist_ref": dist_ref,
                }

        else:
            if self.cfg.TRAIN.use_joints:
                # rs_set = {
                #     "m_ref": feats_ref[:, :min_len, :],
                #     "m_rst": feats_rst[:, :min_len, :],
                #     # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                #     "lat_m": motion_z.permute(1, 0, 2),
                #     "lat_rm": recons_z.permute(1, 0, 2),
                #     "joints_ref": joints_ref,
                #     "joints_rst": joints_rst,
                #     "dist_m": dist_m,
                #     "dist_ref": dist_ref,
                # }
                rs_set["joints_ref"] = joints_ref
                rs_set["joints_rst"] = joints_rst
            else:
                import pdb; pdb.set_trace()
                rs_set = {
                    "m_ref": feats_ref[:, :min_len, :],
                    "m_rst": feats_rst[:, :min_len, :],
                    # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                    "lat_m": motion_z.permute(1, 0, 2),
                    "lat_rm": recons_z.permute(1, 0, 2),
                    "dist_m": dist_m,
                    "dist_ref": dist_ref,
                }

        
        if self.cfg.LOSS.hand_mask:
            rs_set['joint_mask'] = batch['joint_mask'][:, :min_len, :]
            

        if self.cfg.LOSS.Velocity_loss:
            vel_ref = feats_ref[:, :min_len, :][:, 1:, 3:] - feats_ref[:, :min_len, :][:, :-1, 3:]
            vel_rst = feats_rst[:, :min_len, :][:, 1:, 3:] - feats_rst[:, :min_len, :][:, :-1, 3:]
            rs_set['vel_rst'] = vel_rst
            rs_set['vel_ref'] = vel_ref

        # import pdb; pdb.set_trace()

        return rs_set

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist = self.vae.encode(feats_ref, lengths)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_all"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_face"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_body"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['body_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_hand"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['hand_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ['text_face_body']:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ["text_seperate"]:
            
            text = []
            for i in range(len(batch["text"])):
                text.append((batch["text"][i], batch["face_text"][i], batch["body_text"][i], batch["hand_text"][i]))
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                ("", "", "", "") if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            
            semantic_text = []
            face_text = []
            body_text = []
            hand_text = []
            for i in range(len(text)):
                semantic_text.append(text[i][0])
                face_text.append(text[i][1])
                body_text.append(text[i][2])
                hand_text.append(text[i][3])

            cond_emb_semantic = self.text_encoder(semantic_text)
            cond_emb_face = self.text_encoder(face_text)
            cond_emb_body = self.text_encoder(body_text)
            cond_emb_hand = self.text_encoder(hand_text)
            # import pdb; pdb.set_trace()
            cond_emb = self.linear_fusion(cond_emb_semantic, cond_emb_face, cond_emb_body, cond_emb_hand)

        elif self.condition in ["only_pose_concat"]:
            text = []
            for i in range(len(batch["text"])):
                text.append(batch["face_text"][i] +' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ["only_pose_fusion"]:

            text = []
            for i in range(len(batch["text"])):
                text.append((batch["face_text"][i], batch["body_text"][i], batch["hand_text"][i]))
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                ("", "", "") if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            
            face_text = []
            body_text = []
            hand_text = []
            for i in range(len(text)):
                face_text.append(text[i][0])
                body_text.append(text[i][1])
                hand_text.append(text[i][2])

            cond_emb_face = self.text_encoder(face_text)
            cond_emb_body = self.text_encoder(body_text)
            cond_emb_hand = self.text_encoder(hand_text)


            cond_emb = self.linear_fusion(None,cond_emb_face, cond_emb_body, cond_emb_hand)
            # emb_cat = torch.cat((cond_emb_face, cond_emb_body), axis=1)
            # emb_cat = emb_cat.view(emb_cat.size(0), -1)
            # cond_emb = self.emb_fuse(emb_cat).unsqueeze(1)

        
        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        import pdb; pdb.set_trace()
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def t2m_eval(self, batch):
        # import pdb; pdb.set_trace()
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # import pdb; pdb.set_trace()

        if self.vae_type in ["hvq_body_hand_face"]:
            face_ref = batch["face_motion"]
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            # import pdb; pdb.set_trace()
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                _, _, _, _, id_t, id_b = self.vae.encode(motions)
            elif self.vae_type in ["rvq"]:
                quants_1, quants_2 = self.vae.encode(motions)
            elif self.vae_type in ["dual_human_vq"]:
                body_quants, hand_quants = self.vae.encode(motions)
            elif self.vae_type == "hvq_body_hand_face":
                _, _, _, _, _, _, id_f, id_t, id_b = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(id_t, id_b)
            elif self.vae_type in ["hvq_body_hand_face"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(id_f, id_t, id_b)
                face_rst = feats_rst[:, :, -53:]
                feats_rst = feats_rst[:, :, :-53]
                
            elif self.vae_type in ["rvq"]:
                feats_rst = self.vae.forward_decoder(quants_1, quants_2)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["dual_human_vq"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(body_quants, hand_quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("Not supported vae type!")

        # end time
        end = time.time()
        self.times.append(end - start)
        # import pdb; pdb.set_trace()
        # joints recover
        joints_rst = self.feats2joints(feats_rst, self.input_format)
        joints_ref = self.feats2joints(motions, self.input_format)
        
        ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vae_t2m_eval_debug/joints_rst_ori.npy", joints_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vae_t2m_eval_debug/joitns_ref_ori.npy", joints_ref[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        ##################for debug#################

        ############for save tokens#############
        # import pdb; pdb.set_trace()
        # feats_rst = self.renorm2ori(feats_rst)
        # motions = self.renorm2ori(motions)
        # retrieval_name = batch['retrieval_name']
        # feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.TRAIN.DATASETS[0]}_test/{self.vae_type}_VAE_motionx_feats_rst_norm_back_{self.input_format}_joints_hand_ratio_15", retrieval_name[0] + '.npy')
        # feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.TRAIN.DATASETS[0]}_test/{self.vae_type}_VAE_motionx_feats_ref_norm_back_{self.input_format}_joints_hand_ratio_15", retrieval_name[0] + '.npy')

        # feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        # if not os.path.exists(feats_rst_parent_directory):
        #     os.makedirs(feats_rst_parent_directory)

        # feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        # if not os.path.exists(feats_ref_parent_directory):
        #     os.makedirs(feats_ref_parent_directory)
 

        # np.save(feats_rst_path, joints_rst[0].detach().cpu().numpy())
        # np.save(feats_ref_path, joints_ref[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        ############for save tokens#############

        ############for save face#############
        # import pdb; pdb.set_trace()
        # face_rst = self.facerenorm2ori(face_rst)
        # face_ref = self.facerenorm2ori(face_ref)
        # retrieval_name = batch['retrieval_name']
        # feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.TRAIN.DATASETS[0]}_test/{self.vae_type}_VAE_motionx_face_rst", retrieval_name[0] + '.npy')
        # feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.TRAIN.DATASETS[0]}_test/{self.vae_type}_VAE_motionx_face_ref", retrieval_name[0] + '.npy')

        # feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        # if not os.path.exists(feats_rst_parent_directory):
        #     os.makedirs(feats_rst_parent_directory)

        # feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        # if not os.path.exists(feats_ref_parent_directory):
        #     os.makedirs(feats_ref_parent_directory)
 

        # np.save(feats_rst_path, face_rst[0].detach().cpu().numpy())
        # np.save(feats_ref_path, face_ref[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        ############for save face#############

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError

        # ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/joints_ref.npy", joints_ref[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/joints_rst.npy", joints_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/motions.npy", motions[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/feats_rst.npy", feats_rst[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()

        # #################for debug end###############
        
        # if self.cfg.LOSS.Velocity_loss:
        #     import pdb; pdb.set_trace()
        #     vel_ref = motions
        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set
    

    def tmr_t2m_eval(self, batch):
        
        texts = batch["text"]
        texts_ori = copy.deepcopy(batch["text"])
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        name = batch["retrieval_name"]
        
        # import pdb; pdb.set_trace()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            texts_ori = texts_ori * self.cfg.TEST.MM_NUM_REPEATS

            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            
        

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths) # 1, 30 , 256
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                _, _, _, _, id_t, id_b = self.vae.encode(motions)
            elif self.vae_type in ["dual_human_vq"]:
                body_quants, hand_quants = self.vae.encode(motions)
            elif self.vae_type in ["rvq"]:
                quants_1, quants_2 = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths) # 30, 180, 313
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(id_t, id_b)
            elif self.vae_type in ["rvq"]:
                feats_rst = self.vae.forward_decoder(quants_1, quants_2)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["dual_human_vq"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(body_quants, hand_quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("Not supported vae type!")

        # end time
        end = time.time()
        self.times.append(end - start)
        # import pdb; pdb.set_trace()
        # joints recover
        joints_rst = self.feats2joints(feats_rst, self.input_format)
        joints_ref = self.feats2joints(motions, self.input_format)

        # ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/tmr_t2m_eval_debug/joints_rst_mld_0.npy", joints_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/tmr_t2m_eval_debug/joints_ref_mld_0.npy", joints_ref[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        # ##################for debug#################

        #########################saving output###################
        # import pdb; pdb.set_trace()
        # for i in range(joints_rst.shape[0]):
        #     feats_rst = self.renorm2ori(feats_rst)
        #     motions = self.renorm2ori(motions)
        #     feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/mld_icml/feats_rst", name[i] + '.npy')
        #     feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/mld_icml/feats_ref", name[i] + '.npy')
        #     joitns_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/mld_icml/joints_rst", name[i] + '.npy')
        #     joitns_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/mld_icml/joints_ref", name[i] + '.npy')
        #     text_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/mld_icml/text", name[i] + '.txt')

        #     feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        #     if not os.path.exists(feats_rst_parent_directory):
        #         os.makedirs(feats_rst_parent_directory)

        #     feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        #     if not os.path.exists(feats_ref_parent_directory):
        #         os.makedirs(feats_ref_parent_directory)

        #     joints_rst_parent_directory = os.path.dirname(joitns_rst_path)
        #     if not os.path.exists(joints_rst_parent_directory):
        #         os.makedirs(joints_rst_parent_directory)

        #     joints_ref_parent_directory = os.path.dirname(joitns_ref_path)
        #     if not os.path.exists(joints_ref_parent_directory):
        #         os.makedirs(joints_ref_parent_directory)

        #     text_parent_directory = os.path.dirname(text_path)
        #     if not os.path.exists(text_parent_directory):
        #         os.makedirs(text_parent_directory)

        #     np.save(feats_rst_path, feats_rst[i].detach().cpu().numpy())
        #     np.save(feats_ref_path, motions[i].detach().cpu().numpy())
        #     np.save(joitns_rst_path, joints_rst[i].detach().cpu().numpy())
        #     np.save(joitns_ref_path, joints_ref[i].detach().cpu().numpy())
        #     with open(text_path, 'w') as file:
        #         file.write(texts[i] + '\n')
            # import pdb; pdb.set_trace()

        #########################saving output###################

        # renorm for t2m evaluators
        feats_rst_before_renorm4t2m = feats_rst.clone()
        motions_before_renorm4t2m = motions.clone()

        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens_ori = m_lens.clone()
        feats_rst_before_renorm4t2m = feats_rst_before_renorm4t2m[align_idx]
        motions_before_renorm4t2m = motions_before_renorm4t2m[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # import pdb; pdb.set_trace()
        recons_emb_tmr = self.t2m_TMR_motionencoder_eval(feats_rst_before_renorm4t2m, m_lens_ori).loc
        motion_emb_tmr = self.t2m_TMR_motionencoder_eval(motions_before_renorm4t2m, m_lens_ori).loc



        # t2m text encoder
        assert self.cfg.model.eval_text_source in ['caption']


        if self.cfg.model.eval_text_encode_way == 'clip':
            raise NotImplementedError

        elif self.cfg.model.eval_text_encode_way == 't5':
            raise NotImplementedError

        elif 'GRU' in self.cfg.model.eval_text_encode_way:
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx] # 30 ,512
        else:
            raise NotImplementedError

        # if self.trainer.datamodule.is_mm:
        #     import pdb; pdb.set_trace()

        text_emb_tmr = self.t2m_TMR_textencoder_eval(texts_ori).loc[align_idx] # 30 , 256
        # import pdb; pdb.set_trace()

        # ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/joints_ref.npy", joints_ref[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/joints_rst.npy", joints_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/motions.npy", motions[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/vq_vae_t2m_eval_debug/feats_rst.npy", feats_rst[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()

        # #################for debug end###############
        
        # if self.cfg.LOSS.Velocity_loss:
        #     import pdb; pdb.set_trace()
        #     vel_ref = motions

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_t_tmr": text_emb_tmr, 
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "lat_m_tmr": motion_emb_tmr, 
            "lat_rm_tmr": recons_emb_tmr, 
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }


        return rs_set

    def t2m_eval_save_motion_token(self, batch):
        # import pdb; pdb.set_trace()

        name = batch["name"]
        motions = batch["motion"].detach().clone()
        # lengths = batch["length"]
        # word_embs = batch["word_embs"].detach().clone()
        # pos_ohot = batch["pos_ohot"].detach().clone()
        # text_lengths = batch["text_len"].detach().clone()


        
        # start
        start = time.time()

        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.cfg.TEST.MM_NUM_REPEATS
        #     motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
        #     word_embs = word_embs.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        #     pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                           dim=0)
        #     text_lengths = text_lengths.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                _, _, _, _, id_t, id_b = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(id_t, id_b)
            elif self.vae_type in ["rvq"]:
                feats_rst = self.vae.forward_decoder(quants_1, quants_2)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)
        
        # joints recover

        joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.input_format)
        # import pdb; pdb.set_trace()

        joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
        joints_ref = self.feats2joints(motions, skel=self.skel, motion_type=self.input_format)
        joints_ref = joints_ref.view(motions.shape[0], motions.shape[1], self.njoints, 3)




        # import pdb; pdb.set_trace()
        assert len(name) == 1

        feats_rst = self.renorm2ori(feats_rst)
        motions = self.renorm2ori(motions)
        feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.model.vae_type}_VAE_motionx_feats_rst_norm_back", name[0] + '.npy')
        feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.model.vae_type}_VAE_motionx_feats_ref_norm_back", name[0] + '.npy')
        joitns_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.model.vae_type}_VAE_motionx_joints_rst_norm_back", name[0] + '.npy')
        joitns_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/ICML/{self.cfg.model.vae_type}_VAE_motionx_joints_ref_norm_back", name[0] + '.npy')

        feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        if not os.path.exists(feats_rst_parent_directory):
            os.makedirs(feats_rst_parent_directory)

        feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        if not os.path.exists(feats_ref_parent_directory):
            os.makedirs(feats_ref_parent_directory)

        joints_rst_parent_directory = os.path.dirname(joitns_rst_path)
        if not os.path.exists(joints_rst_parent_directory):
            os.makedirs(joints_rst_parent_directory)

        joints_ref_parent_directory = os.path.dirname(joitns_ref_path)
        if not os.path.exists(joints_ref_parent_directory):
            os.makedirs(joints_ref_parent_directory)

        

        np.save(feats_rst_path, feats_rst[0].detach().cpu().numpy())
        np.save(feats_ref_path, motions[0].detach().cpu().numpy())
        np.save(joitns_rst_path, joints_rst[0].detach().cpu().numpy())
        np.save(joitns_ref_path, joints_ref[0].detach().cpu().numpy())

        # ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/feats_rst.npy", feats_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/motion.npy", motions[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/joints_ref_4.npy", joints_ref[4].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/joints_rst_4.npy", joints_rst[4].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        # ##################for debug#################
        
        # import pdb; p
        if self.vae_type in ["hvq", "hvq_body_hand"]:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "motion_code_t": id_t, 
                "motion_code_b": id_b,
                "name": name
            }
        
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "motion_code": quants,
                "name": name
            }
        return rs_set


    def t2m_eval_cal_sort(self, batch):
        import pdb; pdb.set_trace()
        name = batch["name"]
        motions = batch["motion"].detach().clone()
        # lengths = batch["length"]
        # word_embs = batch["word_embs"].detach().clone()
        # pos_ohot = batch["pos_ohot"].detach().clone()
        # text_lengths = batch["text_len"].detach().clone()


        
        # start
        start = time.time()

        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.cfg.TEST.MM_NUM_REPEATS
        #     motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
        #     word_embs = word_embs.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        #     pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                           dim=0)
        #     text_lengths = text_lengths.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                _, _, _, _, id_t, id_b = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(id_t, id_b)
            elif self.vae_type in ["rvq"]:
                feats_rst = self.vae.forward_decoder(quants_1, quants_2)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)
        
        # joints recover

        joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.input_format)
        # import pdb; pdb.set_trace()

        joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
        joints_ref = self.feats2joints(motions, skel=self.skel, motion_type=self.input_format)
        joints_ref = joints_ref.view(motions.shape[0], motions.shape[1], self.njoints, 3)

        feats_rst = self.renorm2ori(feats_rst)
        motions = self.renorm2ori(motions)


        # import pdb; pdb.set_trace()
        assert len(name) == 1

        # feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.model.vae_type}_VAE_motionx_feats_rst_norm_back", name[0] + '.npy')
        # feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.model.vae_type}_VAE_motionx_feats_ref_norm_back", name[0] + '.npy')
        # joitns_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.model.vae_type}_VAE_motionx_joints_rst_norm_back", name[0] + '.npy')
        # joitns_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.model.vae_type}_VAE_motionx_joints_ref_norm_back", name[0] + '.npy')

        # feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        # if not os.path.exists(feats_rst_parent_directory):
        #     os.makedirs(feats_rst_parent_directory)

        # feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        # if not os.path.exists(feats_ref_parent_directory):
        #     os.makedirs(feats_ref_parent_directory)

        # joints_rst_parent_directory = os.path.dirname(joitns_rst_path)
        # if not os.path.exists(joints_rst_parent_directory):
        #     os.makedirs(joints_rst_parent_directory)

        # joints_ref_parent_directory = os.path.dirname(joitns_ref_path)
        # if not os.path.exists(joints_ref_parent_directory):
        #     os.makedirs(joints_ref_parent_directory)

        

        # np.save(feats_rst_path, feats_rst[0].detach().cpu().numpy())
        # np.save(feats_ref_path, motions[0].detach().cpu().numpy())
        # np.save(joitns_rst_path, joints_rst[0].detach().cpu().numpy())
        # np.save(joitns_ref_path, joints_ref[0].detach().cpu().numpy())

        # ##################for debug#################
        # import pdb; pdb.set_trace()
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/feats_rst.npy", feats_rst[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/motion.npy", motions[0].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/joints_ref_4.npy", joints_ref[4].detach().cpu().numpy())
        # np.save("/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_VAE_motionx_debug/joints_rst_4.npy", joints_rst[4].detach().cpu().numpy())
        # import pdb; pdb.set_trace()
        # ##################for debug#################
        
        # import pdb; p
        if self.vae_type in ["hvq", "hvq_body_hand"]:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "motion_code_t": id_t, 
                "motion_code_b": id_b
            }
        
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "motion_code": quants
            }
        return rs_set

    def normal_eval(self, batch):
        # import pdb; pdb.set_trace()
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            elif self.vae_type in ["mld_dual_vae"]:
                body_z, hand_z, body_dist_m, hand_dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["dual_human_vq"]:
                body_quants, hand_quants = self.vae.encode(motions)
            elif self.vae_type in ["rvq"]:
                quants_1, quants_2 = self.vae.encode(motions)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                _, _, _, _, id_t, id_b = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["mld_dual_vae"]:
                feats_rst = self.vae.decode(body_z, hand_z, lengths)
            elif self.vae_type in ["dual_human_vq"]:
                # import pdb; pdb.set_trace()
                feats_rst = self.vae.forward_decoder(body_quants, hand_quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["rvq"]:
                feats_rst = self.vae.forward_decoder(quants_1, quants_2)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type in ["hvq", "hvq_body_hand"]:
                feats_rst = self.vae.forward_decoder(id_t, id_b)
                
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise NotImplenetError

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        # joints_rst = self.feats2joints(feats_rst, self.motion_type)
        # joints_ref = self.feats2joints(motions, self.motion_type)
        # import pdb; pdb.set_trace()




        joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.input_format)
        joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
        joints_ref = self.feats2joints(motions, skel=self.skel, motion_type=self.input_format)
        joints_ref = joints_ref.view(motions.shape[0], motions.shape[1], self.njoints, 3)



        # if self.cfg.TRAIN.use_joints:
        #     rs_set = {
        #         "m_ref": motions,
        #         "m_rst": feats_rst,
        #         "lat_t": text_emb,
        #         "lat_m": motion_emb,
        #         "lat_rm": recons_emb,
        #         "joints_ref": joints_ref,
        #         "joints_rst": joints_rst,
        #     }
        # else:
        #     rs_set = {
        #         "m_ref": motions,
        #         "m_rst": feats_rst,
        #         "lat_t": text_emb,
        #         "lat_m": motion_emb,
        #         "lat_rm": recons_emb,
        #     }

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst, 
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set


    def t2m_eval_smplx(self, batch):
        

        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()
        # import pdb; pdb.set_trace()
        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("Not supported vae type!")

        # end time
        end = time.time()
        self.times.append(end - start)
        # 
        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        #########for check ###########
        # feats_rst = self.renorm2ori(feats_rst)
        # motions = self.renorm2ori(motions)
        # np.save('/comp_robot/lushunlin/visualization/visualization/test_case/mld_vae_check_motion.npy', motions[0].detach().cpu().numpy())
        # np.save('/comp_robot/lushunlin/visualization/visualization/test_case/mld_vae_check_feats.npy', feats_rst[0].detach().cpu().numpy())
        # import pdb; pdb.set_trace()

        #########for check#####################

        #############for save tokens#############
        # import pdb; pdb.set_trace()
        # feats_rst = self.renorm2ori(feats_rst)
        # motions = self.renorm2ori(motions)
        # retrieval_name = batch['retrieval_name']
        # feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.TRAIN.DATASETS[0]}/{self.vae_type}_VAE_motionx_feats_rst_norm_back", retrieval_name[0] + '.npy')
        # feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.TRAIN.DATASETS[0]}/{self.vae_type}_VAE_motionx_feats_ref_norm_back", retrieval_name[0] + '.npy')

        # feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        # if not os.path.exists(feats_rst_parent_directory):
        #     os.makedirs(feats_rst_parent_directory)

        # feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        # if not os.path.exists(feats_ref_parent_directory):
        #     os.makedirs(feats_ref_parent_directory)


        # np.save(feats_rst_path, feats_rst[0].detach().cpu().numpy())
        # np.save(feats_ref_path, motions[0].detach().cpu().numpy())

        #############for save tokens#############

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type in ['smplx_212', 'smplx_159']

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set




    def t2m_eval_smplx_save_motion_token(self, batch):
        # texts = batch["text"]
        name = batch["name"]
        motions = batch["motion"].detach().clone()
        # lengths = batch["length"]
        # word_embs = batch["word_embs"].detach().clone()
        # pos_ohot = batch["pos_ohot"].detach().clone()
        # text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()
        # import pdb; pdb.set_trace()
        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.cfg.TEST.MM_NUM_REPEATS
        #     motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
        #     word_embs = word_embs.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        #     pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                           dim=0)
        #     text_lengths = text_lengths.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq", "spatial_MLP_vqvae", "spatial_transformer_vqvae"]:
                quants = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("Not supported vae type!")

        # end time
        end = time.time()
        self.times.append(end - start)
        # import pdb; pdb.set_trace()
        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)


        
        #########for check ###########
        # feats_rst = self.renorm2ori(feats_rst)
        # motions = self.renorm2ori(motions)
        # np.save('/comp_robot/lushunlin/visualization/visualization/test_case/mld_vae_check_motion.npy', motions[0].detach().cpu().numpy())
        # np.save('/comp_robot/lushunlin/visualization/visualization/test_case/mld_vae_check_feats.npy', feats_rst[0].detach().cpu().numpy())

        #########for check#####################

        import pdb; pdb.set_trace()
        #############for save tokens#############
        feats_rst = self.renorm2ori(feats_rst)
        motions = self.renorm2ori(motions)
        feats_rst_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.TRAIN.DATASETS[0]}/{self.cfg.model.vae_type}_VAE_motionx_feats_rst_norm_back", name[0] + '.npy')
        feats_ref_path = os.path.join(f"/comp_robot/lushunlin/visualization/visualization/test_case/{self.cfg.TRAIN.DATASETS[0]}/{self.cfg.model.vae_type}_VAE_motionx_feats_ref_norm_back", name[0] + '.npy')

        feats_rst_parent_directory = os.path.dirname(feats_rst_path)
        if not os.path.exists(feats_rst_parent_directory):
            os.makedirs(feats_rst_parent_directory)

        feats_ref_parent_directory = os.path.dirname(feats_ref_path)
        if not os.path.exists(feats_ref_parent_directory):
            os.makedirs(feats_ref_parent_directory)


        np.save(feats_rst_path, feats_rst[0].detach().cpu().numpy())
        np.save(feats_ref_path, motions[0].detach().cpu().numpy())

        #############for save tokens#############

        # renorm for t2m evaluators
        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        # motions = self.datamodule.renorm4t2m(motions)
        # # t2m motion encoder
        # m_lens = lengths.copy()
        # m_lens = torch.tensor(m_lens, device=motions.device)
        # align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        # motions = motions[align_idx]
        # feats_rst = feats_rst[align_idx]
        # m_lens = m_lens[align_idx]
        # m_lens = torch.div(m_lens,
        #                    self.cfg.DATASET.HUMANML3D.UNIT_LEN,
        #                    rounding_mode="floor")

        assert self.motion_type == ['smplx_212', 'smplx_159']

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        # recons_mov = self.t2m_moveencoder(feats_rst).detach()
        # recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        # motion_mov = self.t2m_moveencoder(motions).detach()
        # motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        # if self.cfg.model.eval_text_source == 'token':
        #     text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        # elif self.cfg.model.eval_text_source == 'only_text_token':
        #     text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        # elif self.cfg.model.eval_text_source in ['caption']:
        #     if self.cfg.model.eval_text_encode_way == 'clip':
        #         raise NotImplementedError

        #     elif self.cfg.model.eval_text_encode_way == 't5':
        #         raise NotImplementedError

        #     elif 'GRU' in self.cfg.model.eval_text_encode_way:
        #         text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        #     else:
        #         raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                # "lat_t": text_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "motion_code": quants, 
                "name": name
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                # "lat_t": text_emb, 
                "motion_code": quants,
                "name": name
            }
        # import pdb; pdb.set_trace()

        return rs_set


    def t2m_eval_smplx_text_all(self, batch):
        assert self.condition == 'text_all'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_all':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        import pdb; pdb.set_trace()

        return rs_set



    def t2m_eval_smplx_text_face(self, batch):
        assert self.condition == 'text_face'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_face':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set





    def t2m_eval_smplx_text_body(self, batch):
        assert self.condition == 'text_body'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['body_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_body':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set




    def t2m_eval_smplx_text_hand(self, batch):
        assert self.condition == 'text_hand'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['hand_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_hand':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set



    def t2m_eval_smplx_text_face_body(self, batch):
        assert self.condition == 'text_face_body'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_face_body':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set



    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        import pdb; pdb.set_trace()
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # import pdb; pdb.set_trace()
        if split in ["train", "val"]:
            if self.stage == "vae":
                if self.vae_type in ["mld", "vposert","actor"]:
                    rs_set = self.train_vae_forward(batch)
                    # import pdb; pdb.set_trace()
                    rs_set["lat_t"] = rs_set["lat_m"]
                else:
                    rs_set = self.train_vae_forward(batch)
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            
            else:
                raise ValueError(f"Not support this stage {self.stage}!")
            # import pdb; pdb.set_trace()
            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")
            
        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # import pdb; pdb.set_trace()
            if self.condition in ['text', 'text_uncond', 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                # use t2m evaluators
                if self.input_format in ['vector_263', 'root_body_pos_vel_hand_pos_vel']:
                    if self.condition == 'text':
                        if self.cfg.TEST.inference_vq_code:
                            rs_set = self.t2m_eval_save_motion_token(batch)
                        else:
                            if self.cfg.EVAL.use_tmr_eval:
                                # import pdb; pdb.set_trace()
                                rs_set = self.tmr_t2m_eval(batch)
                            else:
                                rs_set = self.t2m_eval(batch)
                    else:
                        raise NotImplementedError
                elif self.input_format in ['smplx_212', 'smplx_159']:
                    if self.condition == 'text':
                        if self.cfg.TEST.inference_vq_code:
                            rs_set = self.t2m_eval_smplx_save_motion_token(batch)
                        else:
                            rs_set = self.t2m_eval_smplx(batch)
                    elif self.condition == 'text_all':
                        rs_set = self.t2m_eval_smplx_text_all(batch)
                    elif self.condition == 'text_face':
                        rs_set = self.t2m_eval_smplx_text_face(batch)
                    elif self.condition == 'text_body':
                        rs_set = self.t2m_eval_smplx_text_body(batch)
                    elif self.condition == 'text_hand':
                        rs_set = self.t2m_eval_smplx_text_hand(batch)
                    elif self.condition == 'text_face_body':
                        rs_set = self.t2m_eval_smplx_text_face_body(batch)
                    else:
                        raise NotImplementedError
                # elif self.input_format in ['root_position', 'root_position_vel', 'root_position_rot6d', 'root_rot6d', 'all', 'root_body_pos_vel_hand_all', 'root_body_pos_vel_hand_pos_vel', 'root_body_pos_vel_hand_pos', 'root_body_pos_vel_hand_rot', 'root_position_vel_only_body', 'root_body_pos_vel_hand_pos_vel_hand_wrist']:
                elif not self.eval_on_text:
                    rs_set = self.normal_eval(batch)
                else:
                    rs_set = self.t2m_eval(batch)
                # else:
                #     raise NotImplementedError
            
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            else:
                raise NotImplementedError
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            # import pdb; pdb.set_trace()
            # metrics_dicts = []
            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                            "motionx",
                            "motionx_v25", 
                            'motionx_v26'
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )
                    # import pdb; pdb.set_trace()
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])

                elif metric == "TemosMetric_body_hand":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                            "motionx",
                            "motionx_v25", 
                            'motionx_v26'
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )
                    # import pdb; pdb.set_trace()
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])

                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "TM2TMetrics_R256":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "TMR_TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t_tmr"],
                        rs_set["lat_rm_tmr"],
                        rs_set["lat_m_tmr"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric in ["MRMetrics", "MRMetrics_body_hand"]:
                    if self.cfg.TEST.inference_vq_code:
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"], 
                                                    rs_set["name"])
                    else:
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
        
                elif metric == "MMMetrics":
                    # import pdb; pdb.set_trace()
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        # self.datamodule.renorm4t2m
        if split in ["test"]:
            # import pdb; pdb.set_trace()
            if self.cfg.TEST.inference_vq_code:
                if self.vae_type in ["hvq", "hvq_body_hand"]:
                    return rs_set["motion_code_t"], rs_set["motion_code_b"], batch["name"]
                else:
                    return rs_set["motion_code"], batch["name"]
            
            if self.motion_type == 'vector_263':
                return rs_set["joints_rst"], batch["length"]
            elif self.motion_type in ['smplx_212', 'smplx_159']:
                if self.cfg.TRAIN.use_joints:
                    # import pdb; pdb.set_trace()
                    return rs_set["m_rst"], batch["length"], rs_set["m_ref"]
                else:
                    return batch["length"]
            elif self.motion_type in ['ric_rot']:
                # import pdb; pdb.set_trace()
                return rs_set["joints_rst"], batch["length"], rs_set["joints_ref"]

            else:
                return batch["length"]
        return loss
