import torch.nn as nn

import torch
import torch.nn.functional as F
import numpy as np

class Decoder_2(nn.Module):
    def __init__(self,
                 top, 
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        
        if top:
            blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            blocks.append(nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(nn.ReLU())

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class QuantizeEMAReset_2(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def dequantize_onehot(self, code_idx):
        # import pdb; pdb.set_trace()
        x = torch.matmul(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        N, width, T = x.shape # 32, 512, 12

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        # commit_loss = F.mse_loss(x, x_d.detach())
        commit_x = x.clone()
        commit_x_d = x_d.clone().detach()
        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, (commit_x, commit_x_d), perplexity, code_idx

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        # import pdb; pdb.set_trace()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)

class Encoder_2(nn.Module):
    def __init__(self,
                 top,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        # import pdb; pdb.set_trace()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if top:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1)) # B, 133, T - > B, 512, T
            blocks.append(nn.ReLU())
            blocks.append(nn.Conv1d(width, width, 4, 2, 1)) # B, 133, T - > B, 512, T
            blocks.append(nn.ReLU())
        else:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1)) # B, 133, T - > B, 512, T
            blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t), # B, 512, T -> B, 512, T/2
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.model(x) # B, joints_feature, T


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
        # import pdb; pdb.set_trace()
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

        x_out, loss_t, loss_b = self.vqvae(x)
        
        return x_out, loss_t, loss_b

    def forward_decoder(self, id_t, id_b, onehot=False):
        # import pdb; pdb.set_trace()
        x_out = self.vqvae.forward_decoder(id_t, id_b, onehot)
        return x_out


if __name__ == '__main__':
    checkpoint_path = '/comp_robot/lushunlin/motion-latent-diffusion/h2vq_motionx_v26_version1_epoch=5999.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    mean = torch.from_numpy(np.load('/comp_robot/lushunlin/motion-latent-diffusion/datasets/Motion-X-V26/mean_std/version1/root_body_pos_vel_hand_pose_vel/mean.npy')).cuda()
    std = torch.from_numpy(np.load('/comp_robot/lushunlin/motion-latent-diffusion/datasets/Motion-X-V26/mean_std/version1/root_body_pos_vel_hand_pose_vel/std.npy')).cuda()
    # import pdb; pdb.set_trace()
    model = HumanVQVAE_2_body_hand(nfeats=313, mu=0.99, quantizer='ema_reset', nb_code=512, \
        code_dim=512, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3, \
            dilation_growth_rate=3, activation='relu', norm=None)
    model.load_state_dict(checkpoint)
    model.eval()
    # x = torch.rand(32, 196, 313).cuda()
    x = torch.from_numpy(np.load('/comp_robot/lushunlin/motion-latent-diffusion/datasets/Motion-X-V26/motion_data/root_body_pos_vel_hand_pos_vel/humanml/000005.npy')).cuda()
    x = x.unsqueeze(0)
    # import pdb; pdb.set_trace()
    x = (x - mean)/ std
    model.cuda()
    with torch.no_grad():
        x_out, loss_t, loss_b = model(x)
    # import pdb; pdb.set_trace()
    print(x_out.shape)
    x_out = x_out * std + mean
    # import pdb; pdb.set_trace()
    id_t = torch.randint(0, 512, (32, 49)).cuda()
    id_b = torch.randint(0, 512, (32, 98)).cuda()
    x_out = model.forward_decoder(id_t, id_b)
    print(x_out.shape)