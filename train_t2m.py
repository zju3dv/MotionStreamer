"""Train original text to motion generation model with llama blocks, Two-Forward strategy and QK-Norm, using the motion latents encoded by the Causal TAE (trained in the first stage)."""

import os
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import json
from accelerate import Accelerator

from models.llama_model import LLaMAHF, LLaMAHFConfig
from humanml3d_272 import dataset_TM_train
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import warnings
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# warm-up + cosine decay scheduler
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters, 
                                                  eta_min=min_lr)
        
    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'total_iters': self.total_iters,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']


args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)


##### ---- Accelerator Setup ---- #####
accelerator = Accelerator()
comp_device = accelerator.device

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


##### ---- Dataloader ---- #####
train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.latent_dir, unit_length=2**args.down_t)


##### ---- Network ---- #####
from sentence_transformers import SentenceTransformer
t5_model = SentenceTransformer('sentencet5-xxl/')
t5_model.eval()
for p in t5_model.parameters():
    p.requires_grad = False


config = LLaMAHFConfig.from_name('Normal_size')
config.block_size = 78
trans_encoder = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    new_ckpt_trans = {}
    for key in ckpt['trans'].keys():
        if key.split('.')[0]=='module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_ckpt_trans[new_key] = ckpt['trans'][key]
    trans_encoder.load_state_dict(new_ckpt_trans, strict=True)
trans_encoder.train()
trans_encoder.to(comp_device)


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter//10, args.total_iter)


t5_model, trans_encoder, optimizer, train_loader = accelerator.prepare(t5_model, trans_encoder, optimizer, train_loader)
train_loader_iter = dataset_TM_train.cycle(train_loader)


diffmlps_batch_mul = 4
def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

import math
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


#--------------2-forward:------------------
def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):

    step = torch.tensor(step, dtype=torch.float32)  
    total_steps = torch.tensor(total_steps, dtype=torch.float32)  
    
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor

def replace_with_pred(latents, pred_xstart, step, total_steps):
    
    decay_factor = cosine_decay(step, total_steps).to(latents.device)
    
    b, l, d = latents.shape
    num_replace = int(l * decay_factor)  
    
    replace_indices = torch.randperm(l)[:num_replace]  

    replace_mask = torch.zeros(b, l, dtype=torch.bool).to(latents.device)
    replace_mask[:, replace_indices] = 1  

    updated_latents = latents.clone()  
    updated_latents[replace_mask] = pred_xstart[replace_mask]
    
    return updated_latents

def forward_loss_withmask_2_forward(latents, trans, m_lens, feat_text, step, total_steps):
    """z: condition; latents: gt"""
    #--------------First Forward:-------------------------
    conditions = trans(latents, feat_text)  
    conditions = conditions.contiguous()
    z = conditions[:,:-1,:]
    #-------------------------------------------------

    b, l, d = latents.shape     
    mask = lengths_to_mask(m_lens, l)       
    mask = mask.reshape(b * l).repeat(diffmlps_batch_mul)

    target = latents.clone().detach()       
    target = target.reshape(b * l, -1)    
    z = z.reshape(b * l, -1)            
    
    with torch.no_grad():
        loss, pred_xstart = trans.diff_loss(target=target, z=z)  

    pred_xstart = pred_xstart.clone().detach()
    pred_xstart = pred_xstart.reshape(b, l, -1)           

    #--------------Second Forward:-------------------------
    # Update latents
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)    
    updated_conditions = trans(updated_latents, feat_text)  
    updated_conditions = updated_conditions.contiguous()
    updated_z = updated_conditions[:,:-1,:]      

    updated_target = latents.clone().detach()       

    updated_target = updated_target.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)    
    updated_z = updated_z.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)            

    updated_target = updated_target[mask]                   
    updated_z = updated_z[mask]                            

    updated_loss, _ = trans.diff_loss(target=updated_target, z=updated_z)  

    return updated_loss
#-------------------

##### ---- Training Loop ---- #####
nb_iter, avg_loss = 0, 0.

while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    text, m_tokens, m_tokens_len = batch
    text = list(text)
    m_tokens, m_tokens_len = m_tokens.to(comp_device), m_tokens_len.to(comp_device)

    bs = len(text)
    num_masked = int(bs * 0.1)  # 10%
    mask_indices = random.sample(range(bs), num_masked)

    for idx in mask_indices:
        text[idx] = ''

    feat_text = torch.from_numpy(t5_model.encode(text)).float()        
    feat_text = feat_text.to(comp_device)

    # -------gt-------- 
    input_latent = m_tokens[:,:-1]    # continuous token
    loss = 0.0

    if args.num_gpus > 1:
        loss = forward_loss_withmask_2_forward(latents=input_latent, trans=trans_encoder.module, m_lens = m_tokens_len, feat_text=feat_text, step=nb_iter, total_steps=args.total_iter)
    else:
        loss = forward_loss_withmask_2_forward(latents=input_latent, trans=trans_encoder, m_lens = m_tokens_len, feat_text=feat_text, step=nb_iter, total_steps=args.total_iter)

    
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step(nb_iter)

    avg_loss = avg_loss + loss.item()

    nb_iter += 1
    args.print_iter = 100
    if nb_iter % args.print_iter ==  0 :
        if accelerator.is_main_process:
            avg_loss = avg_loss / args.print_iter
            writer.add_scalar('./Loss/train', avg_loss, nb_iter)
            writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
            msg = f"Train. Iter {nb_iter} : Loss. {avg_loss:.5f}"
            logger.info(msg)
        avg_loss = 0.


    args.save_iter = 10000
    if nb_iter % args.save_iter == 0:
        # save 
        if accelerator.is_main_process:
            torch.save({
                'trans': trans_encoder.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.out_dir, f'latest.pth'))

                    

accelerator.wait_for_everyone()
