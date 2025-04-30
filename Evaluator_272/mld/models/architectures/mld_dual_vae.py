from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class MldDualVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()
        
        assert nfeats == 313


        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats

        body_input_feats = 4 + 21 * 3 + 22 * 3
        hand_input_feats = 30 * 3 + 30 * 3

        output_feats = nfeats

        body_output_feats = 4 + 21 * 3 + 22 * 3
        hand_output_feats = 30 * 3 + 30 * 3

        self.arch = arch
        self.mlp_dist = ablation.MLP_DIST
        self.pe_type = ablation.PE_TYPE

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            # self.query_pos_encoder = build_position_encoding(
            #     self.latent_dim, position_embedding=position_embedding)
            self.body_query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.hand_query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

            # self.query_pos_decoder = build_position_encoding(
            #     self.latent_dim, position_embedding=position_embedding)
            self.body_query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.hand_query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

        else:
            raise ValueError("Not Support PE type")

        # encoder_layer = TransformerEncoderLayer(
        #     self.latent_dim,
        #     num_heads,
        #     ff_size,
        #     dropout,
        #     activation,
        #     normalize_before,
        # )

        body_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        hand_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        body_encoder_norm = nn.LayerNorm(self.latent_dim)
        hand_encoder_norm = nn.LayerNorm(self.latent_dim)

        # self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
        #                                       encoder_norm)

        self.body_encoder = SkipTransformerEncoder(body_encoder_layer, num_layers,
                                              body_encoder_norm)
        self.hand_encoder = SkipTransformerEncoder(hand_encoder_layer, num_layers,
                                              hand_encoder_norm)


        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":

            body_decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            hand_decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            body_decoder_norm = nn.LayerNorm(self.latent_dim)
            hand_decoder_norm = nn.LayerNorm(self.latent_dim)

            self.body_decoder = SkipTransformerDecoder(body_decoder_layer, num_layers,
                                                  body_decoder_norm)

            self.hand_decoder = SkipTransformerDecoder(hand_decoder_layer, num_layers,
                                                  hand_decoder_norm)


        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            

            self.body_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

            self.hand_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        # self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.body_skel_embedding = nn.Linear(body_output_feats, self.latent_dim)
        self.hand_skel_embedding = nn.Linear(hand_output_feats, self.latent_dim)

        # self.final_layer = nn.Linear(self.latent_dim, output_feats)
        self.body_final_layer = nn.Linear(self.latent_dim, body_output_feats)
        self.hand_final_layer = nn.Linear(self.latent_dim, hand_output_feats)


    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        
        print("Should Not enter here")
        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        body_features = torch.cat((features[..., :4+21*3], features[..., 4+51*3:4+51*3+22*3]), dim=-1)  # (32, 196, 133)
        hand_features = torch.cat((features[..., 4+21*3:4+51*3], features[..., 4+51*3+22*3:]), dim=-1)  # (132, 196, 180)
        bs, nframes, _ = features.shape # (32, 196, 313)
        mask = lengths_to_mask(lengths, device) # (32, 196)

        body_x = body_features
        hand_x = hand_features
        # Embed each human poses into latent vectors
        # x = self.skel_embedding(x)
        body_x = self.body_skel_embedding(body_x)
        hand_x = self.hand_skel_embedding(hand_x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        # x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]  (196, 32, 256)
        body_x = body_x.permute(1,0,2)
        hand_x = hand_x.permute(1,0,2)

        # Each batch has its own set of tokens
        # dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1)) # (2, 32, 256)
        body_dist = torch.tile(self.body_global_motion_token[:, None, :], (1, bs, 1)) # (2, 32, 256)
        hand_dist = torch.tile(self.hand_global_motion_token[:, None, :], (1, bs, 1)) # (2, 32, 256)

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, body_dist.shape[0]),
                                dtype=bool,
                                device=body_x.device) # (32, 2) all one

        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        # xseq = torch.cat((dist, x), 0)
        xseq_body = torch.cat((body_dist, body_x), 0)
        xseq_hand = torch.cat((hand_dist, hand_x), 0)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            # xseq = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq,
            #                     src_key_padding_mask=~aug_mask)[:dist.shape[0]]

            xseq_body = self.body_query_pos_encoder(xseq_body)
            body_dist = self.body_encoder(xseq_body,
                                src_key_padding_mask=~aug_mask)[:body_dist.shape[0]]

            xseq_hand = self.hand_query_pos_encoder(xseq_hand)
            hand_dist = self.hand_encoder(xseq_hand,
                                src_key_padding_mask=~aug_mask)[:hand_dist.shape[0]]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:

            body_mu = body_dist[0:self.latent_size, ...]
            body_logvar = body_dist[self.latent_size:, ...]
            hand_mu = hand_dist[0:self.latent_size, ...]
            hand_logvar = hand_dist[self.latent_size:, ...]


        body_std = body_logvar.exp().pow(0.5)
        body_dist = torch.distributions.Normal(body_mu, body_std)
        body_latent = body_dist.rsample()

        hand_std = hand_logvar.exp().pow(0.5)
        hand_dist = torch.distributions.Normal(hand_mu, hand_std)
        hand_latent = hand_dist.rsample()

        # return latent, dist
        return body_latent, hand_latent, body_dist, hand_dist

    def decode(self, body_z: Tensor, hand_z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, body_z.device)
        bs, nframes = mask.shape

        # queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        body_queries = torch.zeros(nframes, bs, self.latent_dim, device=body_z.device)
        hand_queries = torch.zeros(nframes, bs, self.latent_dim, device=hand_z.device)


        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]


        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                # queries = self.query_pos_decoder(queries)
                body_queries = self.body_query_pos_decoder(body_queries)
                hand_queries = self.hand_query_pos_decoder(hand_queries)
               

                body_output = self.body_decoder(
                    tgt=body_queries,
                    memory=body_z,
                    tgt_key_padding_mask=~mask,

                ).squeeze(0)

                hand_output = self.hand_decoder(
                    tgt=hand_queries,
                    memory=hand_z,
                    tgt_key_padding_mask=~mask,

                ).squeeze(0)


        body_output = self.body_final_layer(body_output)
        hand_output = self.hand_final_layer(hand_output)
        # zero for padded area
        # output[~mask.T] = 0
        body_output[~mask.T] = 0
        hand_output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = torch.cat((body_output.permute(1, 0, 2), hand_output.permute(1, 0, 2)), dim=-1)
        return feats
