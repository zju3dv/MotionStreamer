import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResQuantize(nn.Module):
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
        self.register_buffer('codebook1', torch.zeros(self.nb_code, self.code_dim).cuda())
        self.register_buffer('codebook2', torch.zeros(self.nb_code, self.code_dim).cuda())

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
        self.codebook1 = out[:self.nb_code]
        self.code_sum1 = self.codebook1.clone()
        self.code_count1 = torch.ones(self.nb_code, device=self.codebook1.device)
        self.codebook2 = self.codebook1 - out[:self.nb_code]
        self.code_sum2 = self.codebook2.clone()
        self.code_count2 = torch.ones(self.nb_code, device=self.codebook1.device)
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
    def update_codebook1(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum1 = self.mu * self.code_sum1 + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count1 = self.mu * self.code_count1 + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count1.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum1.view(self.nb_code, self.code_dim) / self.code_count1.view(self.nb_code, 1)

        self.codebook1 = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity
    
    @torch.no_grad()
    def update_codebook2(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum2 = self.mu * self.code_sum2 + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count2 = self.mu * self.code_count2 + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count2.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum2.view(self.nb_code, self.code_dim) / self.code_count2.view(self.nb_code, 1)

        self.codebook2 = usage * code_update + (1 - usage) * code_rand
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
        k_w = self.codebook1.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx1 = torch.min(distance, dim=-1)
        x_1 = F.embedding(code_idx1, self.codebook1)
        x_res = x - x_1
        
        
        k_w = self.codebook2.t()
        distance2 = torch.sum(x_res ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x_res, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx2 = torch.min(distance2, dim=-1)
        
        return code_idx1, code_idx2

    def dequantize(self, code_idx1, code_idx2):
        x1 = F.embedding(code_idx1, self.codebook1)
        x2 = F.embedding(code_idx2, self.codebook2)
        return x1,  x2

    
    def forward(self, x, lambda_code2 = 1.0):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx1, code_idx2 = self.quantize(x)
        x_d1, x_d2 = self.dequantize(code_idx1, code_idx2)

        # Update embeddings
        if self.training:
            perplexity1 = self.update_codebook1(x, code_idx1)
            x_res = x - F.embedding(code_idx1, self.codebook1)
            perplexity2 = self.update_codebook2(x_res, code_idx2)
        else : 
            perplexity1 = self.compute_perplexity(code_idx1)
            perplexity2 = self.compute_perplexity(code_idx2)
        
        # Loss
        # commit_loss = F.mse_loss(x, x_d1.detach()) \
        #               + F.mse_loss(x, x_d1.detach() + x_d2.detach())
        commit_x = x.clone()
        commit_x_d1 = x_d1.clone().detach()
        commit_x_d2 = x_d2.clone().detach()

        # Passthrough
        x_d = x + (x_d1 + x_d2 - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, (commit_x, commit_x_d1, commit_x_d2), (perplexity1, perplexity2)