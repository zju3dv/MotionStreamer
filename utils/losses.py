import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReConsLoss(nn.Module):
    def __init__(self, motion_dim=272):
        super(ReConsLoss, self).__init__()
        self.motion_dim = motion_dim
    
    def softclip(self, tensor, min):
        result_tensor = min + F.softplus(tensor - min)
        return result_tensor
    
    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    
    def forward(self, motion_pred, motion_gt) : 
        """Optimal sigma VAE loss, see https://arxiv.org/pdf/2006.13202 for more details"""
        log_sigma = ((motion_gt[..., :self.motion_dim] - motion_pred[..., :self.motion_dim]) ** 2).mean([0,1,2], keepdim=True).sqrt().log()
        log_sigma = self.softclip(log_sigma, -6)
        loss = self.gaussian_nll(motion_pred[..., :self.motion_dim], log_sigma, motion_gt[..., :self.motion_dim]).sum()
        return loss
    
    
    def forward_KL(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2))
        return loss.mean()
    
    def forward_root(self, motion_pred, motion_gt):
        """[..., :8] relate to the root joint"""
        root_log_sigma = ((motion_gt[..., :8] - motion_pred[..., :8]) ** 2).mean([0,1,2], keepdim=True).sqrt().log()
        root_log_sigma = self.softclip(root_log_sigma, -6)
        root_loss = self.gaussian_nll(motion_pred[..., :8], root_log_sigma, motion_gt[..., :8]).sum()           
        return root_loss
    
