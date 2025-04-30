from typing import List, Optional

import torch
from torch import Tensor
from omegaconf import DictConfig
from mld.models.tools.tools import remove_padding

from mld.models.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from mld.models.modeltype.base import BaseModel
from torch.distributions.distribution import Distribution
from mld.config import instantiate_from_config

from mld.models.losses.temos import TemosLosses
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

from mld.models.architectures import t2m_textenc, t2m_motionenc
import os

import time

import numpy as np
import torch.nn.functional as f
from pathlib import Path

class TEMOS(BaseModel):
    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.is_vae = cfg.model.vae
        self.cfg = cfg
        self.condition = cfg.model.condition
        self.stage = cfg.TRAIN.STAGE
        self.datamodule = datamodule
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.motion_type = cfg.DATASET.MOTION_TYPE

        self.textencoder = instantiate_from_config(cfg.textencoder)
        self.motionencoder = instantiate_from_config(cfg.motionencoder)
        self.motiondecoder = instantiate_from_config(cfg.motiondecoder)


        if self.condition in ["text", "text_uncond", 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
            self.feats2joints = datamodule.feats2joints

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")


        self._losses = MetricCollection({
            split: TemosLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
            for split in ["losses_train", "losses_test", "losses_val"]
        })                   

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        
        if self.cfg.LOSS.USE_INFONCE_FILTER:
            self.filter_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        self.retrieval_text_embedding = []
        self.retrieval_motion_embedding = []
        self.retrieval_sbert_embedding = [] 

        self.retrieval_corres_name = []

        self.gt_idx = 0

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:
        datastruct_from_text = self.text_to_motion_forward(batch["text"],
                                                           batch["length"])

        return remove_padding(datastruct_from_text.joints, batch["length"])


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

        

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )


        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )

        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        
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

        

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.is_vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return features
        return features, latent_vector, distribution

    def motion_to_motion_forward(self, features,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        if self.is_vae:
            distribution = self.motionencoder(features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return features
        return features, latent_vector, distribution


    def save_embeddings(self, batch):
        
        with torch.no_grad():
            motion_all, text_all = None, None
            sbert_embedding_all = None
            
            texts = batch["text"]
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
            word_embs = batch["word_embs"].detach().clone()
            pos_ohot = batch["pos_ohot"].detach().clone()
            text_lengths = batch["text_len"].detach().clone()
            retrieval_name = batch['retrieval_name']
            
            text_embedding = self.textencoder(texts).loc
            motion_embedding = self.motionencoder(motions, lengths).loc

            Emb_text = f.normalize(text_embedding, dim=1)
            Emb_motion = f.normalize(motion_embedding, dim=1)

            if text_all == None:
                text_all = Emb_text
            else:
                text_all = torch.cat((text_all, Emb_text), 0)

            if motion_all == None:
                motion_all = Emb_motion
            else:
                motion_all = torch.cat((motion_all, Emb_motion), 0)

            if self.cfg.LOSS.USE_INFONCE_FILTER:
                sbert_embedding = torch.tensor(self.filter_model.encode(texts)) # (bs, 384)
                sbert_embedding = f.normalize(sbert_embedding, dim=1)

                if sbert_embedding_all == None:
                    sbert_embedding_all = sbert_embedding
                else:
                    sbert_embedding_all = torch.cat((sbert_embedding_all, sbert_embedding), 0)
            
                self.retrieval_sbert_embedding.append(sbert_embedding_all.detach().cpu().numpy())

            self.retrieval_text_embedding.append(text_all.detach().cpu().numpy())
            self.retrieval_motion_embedding.append(motion_all.detach().cpu().numpy())
            self.retrieval_corres_name.append(retrieval_name)
            
            

    def t2m_eval(self, batch):
        retrieval_name = batch['retrieval_name']
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

        assert self.stage in ['temos']

        # Encode the text/decode to a motion
        with torch.no_grad():
            ret = self.text_to_motion_forward(texts,
                                            lengths,
                                            return_latent=True)
            feat_from_text, latent_from_text, distribution_from_text = ret

            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(motions,
                                                lengths,
                                                return_latent=True)
            feat_from_motion, latent_from_motion, distribution_from_motion = ret

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_ref = self.feats2joints(motions)
        joints_rst = self.feats2joints(feat_from_text)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feat_from_text)
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
    

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            # "lat_t": text_emb,
            # "lat_m": motion_emb,
            # "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        
        return rs_set
    

    def tmr_gt_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        # word_embs = batch["word_embs"].detach().clone()
        # pos_ohot = batch["pos_ohot"].detach().clone()
        # text_lengths = batch["text_len"].detach().clone()
        name = batch["retrieval_name"]
        bs, seq = motions.shape[:2]

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
            
            bs = self.cfg.TEST.MM_NUM_REPEATS

        assert self.stage in ['temos']
        self.textencoder.eval()
        self.motionencoder.eval()
        self.motiondecoder.eval()
        with torch.no_grad():

            ret = self.text_to_motion_forward(texts,
                                            lengths,
                                            return_latent=True)
            feat_from_text, latent_from_text, distribution_from_text = ret
            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(motions,
                                                lengths,
                                                return_latent=True)
            feat_from_motion, latent_from_motion, distribution_from_motion = ret
            
            ret = self.motion_to_motion_forward(feat_from_motion, lengths, return_latent=True)
            _, latent_from_motion_rst_motion, _ = ret

        # end time
        end = time.time()
        self.times.append(end - start)
        # joints recover
        joints_ref = self.feats2joints(motions)
        joints_rst = self.feats2joints(feat_from_text)


        # #########################saving output###################
        feats_rst = self.datamodule.renorm4t2m(feat_from_text)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                        self.cfg.DATASET.HUMANML3D_272.UNIT_LEN,
                        rounding_mode="floor")

        recons_emb_tmr = latent_from_motion_rst_motion[align_idx]
        motion_emb_tmr = latent_from_motion[align_idx]
        text_emb_tmr = latent_from_text[align_idx]
        
        self.textencoder.train()
        self.motionencoder.train()
        self.motiondecoder.train()
        
        rs_set = {
            "m_ref": motions,
            "lat_t_tmr": text_emb_tmr, 
            "lat_m_tmr": motion_emb_tmr, 
            "lat_rm_tmr": recons_emb_tmr, 
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        emb_dist = None
        if self.cfg.LOSS.USE_INFONCE and self.cfg.LOSS.USE_INFONCE_FILTER:
            with torch.no_grad():
                text_embedding = self.filter_model.encode(batch["text"])
                text_embedding = torch.tensor(text_embedding).to(batch['motion'][0])
                normalized = f.normalize(text_embedding, p=2, dim=1)
                emb_dist = normalized.matmul(normalized.T)

        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        feat_from_text, latent_from_text, distribution_from_text = ret

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(batch["motion"],
                                            batch["length"],
                                            return_latent=True)
        feat_from_motion, latent_from_motion, distribution_from_motion = ret

        # GT data
        # datastruct_ref = batch["datastruct"]

        # Compare to a Normal distribution
        if self.is_vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        # Compute the losses
        loss = self.losses[split].update(f_text=feat_from_text,
                                         f_motion=feat_from_motion,
                                         f_ref=batch["motion"],
                                         lat_text=latent_from_text,
                                         lat_motion=latent_from_motion,
                                         dis_text=distribution_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_ref=distribution_ref, 
                                         emb_dist=emb_dist)

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

         
        if split in ["val", "test"]:
            # self.save_embeddings(batch)
            if self.cfg.EVAL.eval_self_on_gt:
                rs_set = self.tmr_gt_eval(batch)
            else:
                if self.condition in ['text', 'text_uncond', 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                    # use t2m evaluators
                    rs_set = self.t2m_eval(batch)
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

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit"
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set['lat_t'],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "TMR_TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t_tmr"],
                        rs_set["lat_rm_tmr"],
                        rs_set["lat_m_tmr"],
                        batch["length"],
                    )
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


        if split in ["test"]:
            if self.motion_type == 'vector_263':
                return rs_set["joints_rst"], batch["length"], batch["text"]
            elif self.motion_type == 'smplx_212':
                if self.cfg.TRAIN.use_joints:
                    return rs_set["m_rst"], batch["length"], rs_set["m_ref"]
                else:
                    return batch["length"]

        return loss


    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["val", "test"]:
            
            if (self.trainer.current_epoch+1) % 1000 == 0:
                output_dir = Path(
                    os.path.join(
                        self.cfg.FOLDER,
                        str(self.cfg.model.model_type),
                        str(self.cfg.NAME),
                        "embeddings",
                        split,
                        "epoch_" + str(self.trainer.current_epoch)
                    ))
                
                os.makedirs(output_dir, exist_ok=True)
                
                self.retrieval_text_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_text_embedding)], dim=0)
                self.retrieval_motion_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_motion_embedding)], dim=0)
                

                tmp_retrieval_name = []
                for i in self.all_gather(self.retrieval_corres_name):
                    tmp_retrieval_name += i
                self.retrieval_corres_name = tmp_retrieval_name
                with open(output_dir/"test_name_debug.txt", "w") as test_name_file:
                    for i in self.retrieval_corres_name:
                        test_name_file.write(i + '\n')
                
                if self.cfg.LOSS.USE_INFONCE_FILTER:
                    self.retrieval_sbert_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_sbert_embedding)], dim=0)
                    np.save(output_dir/"sbert_embedding.npy", self.retrieval_sbert_embedding.detach().cpu().numpy())

       
                np.save(output_dir/"text_embedding.npy", self.retrieval_text_embedding.detach().cpu().numpy())# (2324, 256)
                np.save(output_dir/"motion_embedding.npy", self.retrieval_motion_embedding.detach().cpu().numpy())

                print('save embedding in {} at {}'.format(output_dir, self.trainer.current_epoch))
                
            
            self.retrieval_text_embedding = []
            self.retrieval_motion_embedding = []
            self.retrieval_sbert_embedding = []

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val", "test"]:

            if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
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
