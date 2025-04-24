import torch.nn as nn
from .encdec import Encoder, Decoder, Spatial_MLP_Encoder, Spatial_MLP_Decoder, Spatial_transformer_Encoder, Spatial_transformer_Decoder, Encoder_2, Decoder_2
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, QuantizeEMAReset_2
from .res_quantize import ResQuantize
import torch


class VQVAE_251(nn.Module):
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
        # import pdb; pdb.set_trace()
        dim_input = nfeats  
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")
        self.encoder = Encoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)
        elif quantizer == "residual_ema_reset":
            self.quantizer = ResQuantize(nb_code, code_dim, mu)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        # import pdb; pdb.set_trace()
        N, T, _ = x.shape
        x_in = self.preprocess(x) # 32， 313， 196
        x_encoder = self.encoder(x_in) # 32, 512, 49
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # import pdb; pdb.set_trace()
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

class VQVAE_2_251(nn.Module):
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
        # self.quant = quantizer

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")

        # dim_input 313
        # output_emb_width 512
        # import pdb; pdb.set_trace()
        self.encoder_b = Encoder_2(False, dim_input, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_t = Encoder_2(False, output_emb_width, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantize_conv_t = nn.Conv1d(output_emb_width, output_emb_width, 1)

        self.quantize_t = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder_t = Decoder_2(False, output_emb_width, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.quantize_conv_b = nn.Conv1d(output_emb_width + output_emb_width, output_emb_width, 1)

        self.quantize_b = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder = Decoder_2(False, dim_input, output_emb_width+output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.upsample_t = nn.ConvTranspose1d(
            output_emb_width, output_emb_width, 4, stride=2, padding=1
        )

        # self.dec = 

        # if quantizer == "ema_reset":
        #     self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        # elif quantizer == "orig":
        #     self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        # elif quantizer == "ema":
        #     self.quantizer = QuantizeEMA(nb_code, code_dim)
        # elif quantizer == "reset":
        #     self.quantizer = QuantizeReset(nb_code, code_dim)
        # elif quantizer == "residual_ema_reset":
        #     self.quantizer = ResQuantize(nb_code, code_dim, mu)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        
        N, T, _ = x.shape # 32, 196, 313
        x_in = self.preprocess(x) # 32, 313, 196
        # x_encoder = self.encoder(x_in)
        x_encoder_b = self.encoder_b(x_in) # 32, 512, 24
        # 
        x_encoder_t = self.encoder_t(x_encoder_b) # 32, 512, 12
        # import pdb; pdb.set_trace()
        quant_t = self.quantize_conv_t(x_encoder_t) # 32, 512 ,12

        # x_encoder_b = self.postprocess(x_encoder_b) # 32, 24, 512
        # quant_t = self.postprocess(quant_t) # 32, 12, 512

        
        # quant_t = quant_t.contiguous().view(-1, quant_t.shape[-1]) # 384, 512
        quant_t, loss_t, _, id_t = self.quantize_t(quant_t) #quant_t 32, 512, 12, id_t = 384
        # import pdb; pdb.set_trace()
        dec_t = self.decoder_t(quant_t) # 32, 512, 24

        x_encoder_b = torch.cat([dec_t, x_encoder_b], 1)
        quant_b = self.quantize_conv_b(x_encoder_b) # 32, 512, 24
        # quant_b = quant_b.contiguous().view(-1, quant_b.shape[-1])
        quant_b, loss_b, _, id_b = self.quantize_b(quant_b) # quant_b 32, 512, 24
        
        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        # code_idx = self.quantizer.quantize(x_encoder)
        id_t = id_t.view(N, -1) # 32, 12
        id_b = id_b.view(N, -1) # 32, 24
        return quant_t, quant_b, loss_t, loss_b, id_t, id_b


    def forward(self, x):
        # x (32, 196, 313)
        # import pdb; pdb.set_trace()
        # x_in = self.preprocess(x) # (32, 313, 196)
        # # Encode
        # x_encoder = self.encoder(x_in)
        
        # ## quantization
        # x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        quant_t, quant_b, loss_t, loss_b, id_t, id_b = self.encode(x)

        ## decoder
        x_decoder = self.decode(quant_t, quant_b)
        x_out = self.postprocess(x_decoder)
        return x_out, loss_t, loss_b

    def decode(self, quant_t, quant_b):
        '''
        Input:
        quant_t shape [32, 512, 49]
        quant_b shape [32, 512, 98]
        Output:
        dec shape [32, 313, 196]
        '''
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1) # 32, 1024, 98
        # import pdb; pdb.set_trace()
        dec = self.decoder(quant)
        return dec


    def forward_decoder(self, id_t, id_b):
        # import pdb; pdb.set_trace()
        x_d_t = self.quantize_t.dequantize(id_t) # 
        x_d_t = x_d_t.view(id_t.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        x_d_b = self.quantize_b.dequantize(id_b)
        x_d_b = x_d_b.view(id_b.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        
        dec = self.decode(x_d_t, x_d_b)
        # decoder
        x_out = self.postprocess(dec)
        # import pdb; pdb.set_trace()
        return x_out



class VQVAE_2_251_body_hand(nn.Module):
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
        # self.quant = quantizer
        # import pdb; pdb.set_trace()
        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")

        # dim_input 313
        # output_emb_width 512
        # import pdb; pdb.set_trace()
        self.encoder_body = Encoder_2(False, 133, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_hand = Encoder_2(False, 180, output_emb_width, 2, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantize_conv_h = nn.Conv1d(output_emb_width, output_emb_width, 1)

        self.quantize_body = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder_hand = Decoder_2(False, output_emb_width, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.quantize_conv_b = nn.Conv1d(output_emb_width + output_emb_width, output_emb_width, 1)

        self.quantize_hand = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder = Decoder_2(False, dim_input, output_emb_width+output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.upsample_t = nn.ConvTranspose1d(
            output_emb_width, output_emb_width, 4, stride=2, padding=1
        )

        # self.dec = 

        # if quantizer == "ema_reset":
        #     self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        # elif quantizer == "orig":
        #     self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        # elif quantizer == "ema":
        #     self.quantizer = QuantizeEMA(nb_code, code_dim)
        # elif quantizer == "reset":
        #     self.quantizer = QuantizeReset(nb_code, code_dim)
        # elif quantizer == "residual_ema_reset":
        #     self.quantizer = ResQuantize(nb_code, code_dim, mu)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        



        N, T, _ = x.shape # 32, 196, 313
        x_in = self.preprocess(x) # 32, 313, 196
        # x_encoder = self.encoder(x_in)
        body_x_in = torch.cat((x_in[:, :4+21*3, :], x_in[:, 4+51*3:4+51*3+22*3, :]), dim=-2)  # (32, 133, 196)
        hand_x_in = torch.cat((x_in[:, 4+21*3:4+51*3, :], x_in[:, 4+51*3+22*3:, :]), dim=-2)  # (132, 180, 196)

        # x_encoder_b = self.encoder_body(body_x_in) # 32, 512, 98
        # x_encoder_b = self.encoder_body(x_in)
        x_encoder_b = self.encoder_body(body_x_in) # 32, 512, 98

        # x_encoder_h = self.encoder_hand(hand_x_in) # 32, 512, 49
        x_encoder_h = self.encoder_hand(hand_x_in) # 32, 512, 49
        # import pdb; pdb.set_trace()
        quant_h = self.quantize_conv_h(x_encoder_h) # 32, 512 ,49

        # x_encoder_b = self.postprocess(x_encoder_b) # 32, 24, 512
        # quant_t = self.postprocess(quant_t) # 32, 12, 512

        
        # quant_t = quant_t.contiguous().view(-1, quant_t.shape[-1]) # 384, 512
        quant_hand, loss_hand, _, id_hand = self.quantize_hand(quant_h) #quant_hand 32, 512, 49, id_hand = 1568
        # import pdb; pdb.set_trace()
        dec_hand = self.decoder_hand(quant_hand) # 32, 512, 98
        # import pdb; pdb.set_trace()
        x_encoder_b = torch.cat([dec_hand, x_encoder_b], 1)
        # x_encoder_b = torch.cat([x_encoder_b, dec_hand], 1)
        quant_b = self.quantize_conv_b(x_encoder_b) # 32, 512, 98
        # quant_b = quant_b.contiguous().view(-1, quant_b.shape[-1])
        quant_b, loss_b, _, id_b = self.quantize_body(quant_b) # quant_b 32, 512, 98 quant_b = 3136
        
        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        # code_idx = self.quantizer.quantize(x_encoder)
        id_hand = id_hand.view(N, -1) # 32, 12
        id_b = id_b.view(N, -1) # 32, 24
        return quant_hand, quant_b, loss_hand, loss_b, id_hand, id_b


    def forward(self, x):
        # x (32, 196, 313)
        # import pdb; pdb.set_trace()
        # x_in = self.preprocess(x) # (32, 313, 196)
        # # Encode
        # x_encoder = self.encoder(x_in)
        
        # ## quantization
        # x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        quant_t, quant_b, loss_t, loss_b, id_t, id_b = self.encode(x)

        ## decoder
        x_decoder = self.decode(quant_t, quant_b)
        x_out = self.postprocess(x_decoder)
        return x_out, loss_t, loss_b

    def decode(self, quant_t, quant_b):
        '''
        Input:
        quant_t shape [32, 512, 49]
        quant_b shape [32, 512, 98]
        Output:
        dec shape [32, 313, 196]
        '''
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1) # 32, 1024, 98
        # import pdb; pdb.set_trace()
        dec = self.decoder(quant)
        return dec


    def forward_decoder(self, id_t, id_b, onehot=False):
        '''
        id_t: (1, 49)
        id_b: (1, 98)
        '''
        # import pdb; pdb.set_trace()
        if onehot:
            x_d_t = self.quantize_hand.dequantize_onehot(id_t) # 
            x_d_b = self.quantize_body.dequantize_onehot(id_b)
        else:
            x_d_t = self.quantize_hand.dequantize(id_t) # 
            x_d_b = self.quantize_body.dequantize(id_b)

        # import pdb; pdb.set_trace()
        x_d_t = x_d_t.view(id_t.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous() # (1, 512, 49)
        x_d_b = x_d_b.view(id_b.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        
        dec = self.decode(x_d_t, x_d_b)
        # decoder
        x_out = self.postprocess(dec)
        # import pdb; pdb.set_trace()
        return x_out
    

class VQVAE_2_251_body_hand_face(nn.Module):
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
        # self.quant = quantizer

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")

        # dim_input 313
        # output_emb_width 512
        # import pdb; pdb.set_trace()
        self.encoder_body = Encoder_2(False, 133, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_hand = Encoder_2(False, 180, output_emb_width, 2, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_face = Encoder_2(False, 53, output_emb_width, 2, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)


        self.quantize_conv_h = nn.Conv1d(output_emb_width, output_emb_width, 1)

        self.quantize_conv_f = nn.Conv1d(output_emb_width, output_emb_width, 1)

        self.quantize_body = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder_hand = Decoder_2(False, output_emb_width, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder_face = Decoder_2(False, output_emb_width, output_emb_width, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.quantize_conv_b = nn.Conv1d(output_emb_width * 3, output_emb_width, 1)

        self.quantize_hand = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.quantize_face = QuantizeEMAReset_2(nb_code, code_dim, mu)

        self.decoder = Decoder_2(False, dim_input+53, output_emb_width*3, 1, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.upsample_t = nn.ConvTranspose1d(
            output_emb_width, output_emb_width, 4, stride=2, padding=1
        )

        self.upsample_f = nn.ConvTranspose1d(
            output_emb_width, output_emb_width, 4, stride=2, padding=1
        )

        # self.dec = 

        # if quantizer == "ema_reset":
        #     self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        # elif quantizer == "orig":
        #     self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        # elif quantizer == "ema":
        #     self.quantizer = QuantizeEMA(nb_code, code_dim)
        # elif quantizer == "reset":
        #     self.quantizer = QuantizeReset(nb_code, code_dim)
        # elif quantizer == "residual_ema_reset":
        #     self.quantizer = ResQuantize(nb_code, code_dim, mu)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        
        N, T, _ = x.shape # 32, 196, 366
        x_in = self.preprocess(x) # 32, 366, 196
        # x_encoder = self.encoder(x_in)
        body_x_in = torch.cat((x_in[:, :4+21*3, :], x_in[:, 4+51*3:4+51*3+22*3, :]), dim=-2)  # (32, 133, 80)
        hand_x_in = torch.cat((x_in[:, 4+21*3:4+51*3, :], x_in[:, 4+51*3+22*3:4+51*3+52*3, :]), dim=-2)  # (132, 180, 80)
        face_x_in = x_in[:, -53:, :]  # (132, 53, 80)
        # import pdb; pdb.set_trace()
        # x_encoder_b = self.encoder_body(body_x_in) # 32, 512, 80
        # x_encoder_b = self.encoder_body(x_in)
        x_encoder_b = self.encoder_body(body_x_in) # 32, 512, 40
    
        # x_encoder_h = self.encoder_hand(hand_x_in) # 32, 512, 49
        x_encoder_h = self.encoder_hand(hand_x_in) # 32, 512, 40

        x_encoder_f = self.encoder_face(face_x_in) # 32, 512, 40
        
        # import pdb; pdb.set_trace()
        quant_h = self.quantize_conv_h(x_encoder_h) # 32, 512 ,40


        # x_encoder_b = self.postprocess(x_encoder_b) # 32, 24, 512
        # quant_t = self.postprocess(quant_t) # 32, 12, 512

        
        # quant_t = quant_t.contiguous().view(-1, quant_t.shape[-1]) # 384, 512
        quant_hand, loss_hand, _, id_hand = self.quantize_hand(quant_h) #quant_hand 32, 512, 40, id_hand = 1568
        
        # import pdb; pdb.set_trace()
        dec_hand = self.decoder_hand(quant_hand) # 32, 512, 80

        quantize_f = self.quantize_conv_f(x_encoder_f) # 32, 512, 40
        quant_face, loss_face, _, id_face = self.quantize_face(quantize_f) # quant_face 32, 512, 40, id_face = 1568
        dec_face = self.decoder_face(quant_face) # 32, 512, 80
        
        
        x_encoder_b = torch.cat([dec_face, dec_hand, x_encoder_b], 1) # 32, 1536, 40
        # x_encoder_b = torch.cat([x_encoder_b, dec_hand], 1)
        quant_b = self.quantize_conv_b(x_encoder_b) # 32, 512, 40
        # import pdb; pdb.set_trace()
        # quant_b = quant_b.contiguous().view(-1, quant_b.shape[-1])
        quant_b, loss_b, _, id_b = self.quantize_body(quant_b) # quant_b 32, 512, 40 quant_b = 3136
        
        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        # code_idx = self.quantizer.quantize(x_encoder)
        id_hand = id_hand.view(N, -1) # 32, 20
        id_face = id_face.view(N, -1) # 32, 20
        id_b = id_b.view(N, -1) # 32, 40
        return quant_face, quant_hand, quant_b, loss_face, loss_hand, loss_b, id_face, id_hand, id_b


    def forward(self, x):
        # x (32, 196, 313)
        # import pdb; pdb.set_trace()
        # x_in = self.preprocess(x) # (32, 313, 196)
        # # Encode
        # x_encoder = self.encoder(x_in)
        
        # ## quantization
        # x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        quant_f, quant_h, quant_b, loss_f, loss_h, loss_b, id_f, id_h, id_b = self.encode(x)

        ## decoder
        x_decoder = self.decode(quant_f, quant_h, quant_b)
        # import pdb; pdb.set_trace()
        x_out = self.postprocess(x_decoder)
        return x_out, loss_f, loss_h, loss_b

    def decode(self, quant_f, quant_h, quant_b):
        '''
        Input:
        quant_f shape [32, 512, 20]
        quant_t shape [32, 512, 20]
        quant_b shape [32, 512, 40]
        Output:
        dec shape [32, 366, 80]
        '''
        
        upsample_h = self.upsample_t(quant_h)

        upsample_f = self.upsample_f(quant_f)

        quant = torch.cat([upsample_f, upsample_h, quant_b], 1) # 32, 1024, 40
        # 
        dec = self.decoder(quant)
        # import pdb; pdb.set_trace()
        return dec


    def forward_decoder(self, id_f, id_t, id_b, onehot=False):
        '''
        id_t: (1, 49)
        id_b: (1, 98)
        '''
        # import pdb; pdb.set_trace()
        if onehot:
            x_d_f = self.quantize_face.dequantize_onehot(id_f)
            x_d_t = self.quantize_hand.dequantize_onehot(id_t) # 
            x_d_b = self.quantize_body.dequantize_onehot(id_b)
        else:
            x_d_f = self.quantize_face.dequantize(id_f)
            x_d_t = self.quantize_hand.dequantize(id_t) # 
            x_d_b = self.quantize_body.dequantize(id_b)

        # import pdb; pdb.set_trace()
        x_d_f = x_d_f.view(id_f.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_d_t = x_d_t.view(id_t.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous() # (1, 512, 49)
        x_d_b = x_d_b.view(id_b.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        
        dec = self.decode(x_d_f, x_d_t, x_d_b)
        # decoder
        x_out = self.postprocess(dec)
        # import pdb; pdb.set_trace()
        return x_out



class HumanVQVAE_2_body_hand(nn.Module):
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
        # import pdb; pdb.set_trace()
        self.vqvae = VQVAE_2_251_body_hand(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss_f, loss_t, loss_b = self.vqvae(x)
        
        return x_out, loss_f, loss_t, loss_b

    def forward_decoder(self, id_t, id_b, onehot=False):
        # import pdb; pdb.set_trace()
        x_out = self.vqvae.forward_decoder(id_t, id_b, onehot)
        return x_out
    

class HumanVQVAE_2_body_hand_face(nn.Module):
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
        # import pdb; pdb.set_trace()
        self.vqvae = VQVAE_2_251_body_hand_face(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss_t, loss_b = self.vqvae(x)
        
        return x_out, loss_t, loss_b

    def forward_decoder(self, id_f, id_t, id_b, onehot=False):
        x_out = self.vqvae.forward_decoder(id_f, id_t, id_b, onehot)
        return x_out

class HumanVQVAE_2(nn.Module):
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
        self.vqvae = VQVAE_2_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss_t, loss_b = self.vqvae(x)
        
        return x_out, loss_t, loss_b

    def forward_decoder(self, id_t, id_b, onehot=False):
        x_out = self.vqvae.forward_decoder(id_t, id_b, onehot=False)
        return x_out


class HumanVQVAE(nn.Module):
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
        # import pdb; pdb.set_trace()
        # self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, (commit_x, commit_x_d), perplexity = self.vqvae(x)
        
        return x_out, (commit_x, commit_x_d), perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
    





class Residual_VQVAE_251(nn.Module):
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

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")
        self.encoder = Encoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        assert quantizer == "residual_ema_reset"

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)
        elif quantizer == "residual_ema_reset":
            self.quantizer = ResQuantize(nb_code, code_dim, mu)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        # import pdb; pdb.set_trace()
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx1, code_idx2 = self.quantizer.quantize(x_encoder)
        code_idx1 = code_idx1.view(N, -1)
        code_idx2 = code_idx2.view(N, -1)
        return code_idx1, code_idx2


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, code_idx1, code_idx2):
        x_d1, x_d2 = self.quantizer.dequantize(code_idx1, code_idx2)
        x_d = x_d1 + x_d2
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class Human_Residual_VQVAE(nn.Module):
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
        self.vqvae = Residual_VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, (commit_x, commit_x_d1, commit_x_d2), perplexity = self.vqvae(x)
        
        return x_out, (commit_x, commit_x_d1, commit_x_d2), perplexity

    def forward_decoder(self, quants1, quants2):
        # import pdb; pdb.set_trace()
        x_out = self.vqvae.forward_decoder(quants1, quants2)
        return x_out
        

class Spatial_MLP_VQVAE_251(nn.Module):
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

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")
        self.encoder = Spatial_MLP_Encoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Spatial_MLP_Decoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
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
        # import pdb; pdb.set_trace()
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class Spatial_MLP_HumanVQVAE(nn.Module):
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
        self.vqvae = Spatial_MLP_VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, (commit_x, commit_x_d), perplexity = self.vqvae(x)
        
        return x_out, (commit_x, commit_x_d), perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
    






class Spatial_transformer_VQVAE_251(nn.Module):
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

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")
        self.encoder = Spatial_transformer_Encoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Spatial_transformer_Decoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
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
        # import pdb; pdb.set_trace()
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class Spatial_transformer_HumanVQVAE(nn.Module):
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
        self.vqvae = Spatial_transformer_VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # import pdb; pdb.set_trace()
    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, (commit_x, commit_x_d), perplexity = self.vqvae(x)
        
        return x_out, (commit_x, commit_x_d), perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out