import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
import torch


class dual_VQVAE_251(nn.Module):
    def __init__(self,
                 nfeats: int,
                 mu, 
                 quantizer='ema_reset', 
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer

        # dim_input = nfeats
        assert nfeats == 313
        body_dim_input =  4 + 21 * 3 + 22 * 3 
        hand_dim_input = 30 * 3 + 30 * 3

        self.body_encoder = Encoder(body_dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.hand_encoder = Encoder(hand_dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        self.body_decoder = Decoder(body_dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.hand_decoder = Decoder(hand_dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        if quantizer == "ema_reset":
            self.body_quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
            self.hand_quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        body_x_in = torch.cat((x_in[:, :4+21*3, :], x_in[:, 4+51*3:4+51*3+22*3, :]), dim=-2)  # (32, 133, 196)
        hand_x_in = torch.cat((x_in[:, 4+21*3:4+51*3, :], x_in[:, 4+51*3+22*3:, :]), dim=-2)  # (132, 180, 196)
        # x_encoder = self.encoder(x_in)
        body_x_encoder = self.body_encoder(body_x_in)
        hand_x_encoder = self.hand_encoder(hand_x_in)

        body_x_encoder = self.postprocess(body_x_encoder)
        hand_x_encoder = self.postprocess(hand_x_encoder)

        body_x_encoder = body_x_encoder.contiguous().view(-1, body_x_encoder.shape[-1])  # (NT, C)
        hand_x_encoder = hand_x_encoder.contiguous().view(-1, hand_x_encoder.shape[-1])  # (NT, C)

        body_code_idx = self.body_quantizer.quantize(body_x_encoder)
        hand_code_idx = self.hand_quantizer.quantize(hand_x_encoder)

        body_code_idx = body_code_idx.view(N, -1)
        hand_code_idx = hand_code_idx.view(N, -1)

        return body_code_idx, hand_code_idx


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_in = self.preprocess(x)
        body_x_in = torch.cat((x_in[:, :4+21*3, :], x_in[:, 4+51*3:4+51*3+22*3, :]), dim=-2)  # (32, 133, 196)
        hand_x_in = torch.cat((x_in[:, 4+21*3:4+51*3, :], x_in[:, 4+51*3+22*3:, :]), dim=-2)  # (132, 180, 196)

        # Encode
        body_x_encoder = self.body_encoder(body_x_in)
        hand_x_encoder = self.hand_encoder(hand_x_in)
        
        ## quantization
        body_x_quantized, body_loss, body_perplexity  = self.body_quantizer(body_x_encoder)
        hand_x_quantized, hand_loss, hand_perplexity  = self.hand_quantizer(hand_x_encoder)

        ## decoder
        body_x_decoder = self.body_decoder(body_x_quantized)
        hand_x_decoder = self.hand_decoder(hand_x_quantized)
        
        body_x_out = self.postprocess(body_x_decoder)
        hand_x_out = self.postprocess(hand_x_decoder)

        # import pdb; pdb.set_trace()
        x_out = torch.cat((body_x_out, hand_x_out), dim=-1)
        return x_out, body_loss, hand_loss, body_perplexity, hand_perplexity


    def forward_decoder(self, body_quants, hand_quants):
        
        body_x_d = self.body_quantizer.dequantize(body_quants)
        hand_x_d = self.hand_quantizer.dequantize(hand_quants)

        body_x_d = body_x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        hand_x_d = hand_x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        body_x_decoder = self.body_decoder(body_x_d)
        hand_x_decoder = self.hand_decoder(hand_x_d)

        body_x_out = self.postprocess(body_x_decoder)
        hand_x_out = self.postprocess(hand_x_decoder)
        
        x_out = torch.cat((body_x_out, hand_x_out), dim=-1)
        return x_out



class DualHumanVQVAE(nn.Module):
    def __init__(self,
                 nfeats: int, 
                 mu, 
                 quantizer='ema_reset', 
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 **kwargs):
        
        super().__init__()
        # self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = dual_VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        body_quants, hand_quants = self.vqvae.encode(x) # (N, T)
        return body_quants, hand_quants

    def forward(self, x):

        x_out, (body_commit_x, body_commit_x_d), (hand_commit_x, hand_commit_x_d), body_perplexity, hand_perplexity = self.vqvae(x)
        
        return x_out, (body_commit_x, body_commit_x_d), (hand_commit_x, hand_commit_x_d), body_perplexity, hand_perplexity

    def forward_decoder(self, body_quants, hand_quants):
        x_out = self.vqvae.forward_decoder(body_quants, hand_quants)
        return x_out
        