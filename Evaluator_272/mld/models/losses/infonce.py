import torch
import torch.nn.functional as F
import numpy as np

class InfoNCE:
    def __init__(self, t):
        # pass
        self.t = t

    def __call__(self, f, dist):
        '''
        f_motion: N x d 
        f_text: N x d 
        '''
        t = self.t
        f_motion, f_text = f[0], f[1]
        N, d = f_motion.shape[0], f_motion.shape[1]

        
        Emb_motion = F.normalize(f_motion, dim=1)
        Emb_text = F.normalize(f_text, dim=1)
        
        t = torch.tensor(t).to(f_motion.device)
        logits = torch.mm(Emb_motion, Emb_text.T)
        # logits = torch.mm(Emb_motion, Emb_text.T) / torch.exp(t)
        if dist is not None:
            text_logits = dist.detach()
            mask = torch.where(torch.logical_and(text_logits > 0.85, text_logits < 1.0-1e-100), torch.tensor(float('-inf')).to(f_motion.device), torch.tensor(1.0e100).to(f_motion.device))
            mask.diagonal().fill_(float('inf'))
            logits = torch.min(mask, logits)
            # mask = torch.where((torch.logical_and(text_logits > 0.985, text_logits < 1.0-1e-100)), torch.tensor(float('-inf')).cuda(), torch.tensor(1.0e100).cuda())
            # logits = torch.min(mask, logits)
        
        N = f_motion.shape[0]
        labels = torch.arange(N).to(f_motion.device)
        
        loss_m = F.cross_entropy(logits / t, labels)
        loss_t = F.cross_entropy(logits.T / t, labels)

        loss = (loss_m + loss_t) / 2
        
        return loss
        
    def __repr__(self):
        return "InfoNCE()"