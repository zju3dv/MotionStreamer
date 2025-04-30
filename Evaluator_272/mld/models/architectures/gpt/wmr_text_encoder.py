import os
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from mld.models.operator import PositionalEncoding
from mld.utils.temos_utils import lengths_to_mask

from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from collections import OrderedDict
import pytorch_lightning as pl

class TextEncoder(pl.LightningModule):

    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        last_hidden_state: bool = False,
        latent_dim: list = [1, 256],
    ) -> None:

        super().__init__()

        self.latent_dim = latent_dim

        model_dict = OrderedDict()
        state_dict = torch.load(modelpath)["state_dict"]

        self.text_model = DistilbertActorAgnosticEncoder('distilbert-base-uncased', num_layers=4)

        for k, v in state_dict.items():
            # print(k)
            if k.split(".")[0] == "textencoder":
                name = k.replace("textencoder.", "")
                model_dict[name] = v

        self.text_model.load_state_dict(model_dict, strict=True)

        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        

    def forward(self, texts: List[str]):
        feat_clip_text = self.text_model(texts).loc.to(self.text_model.device)
        feat_clip_text = torch.cat((feat_clip_text, feat_clip_text), dim=1)
    
        return feat_clip_text
