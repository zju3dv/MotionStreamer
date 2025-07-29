import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import models.tae as tae
import utils.losses as losses 
import options.option_tae as option_tae
import utils.utils_model as utils_model
from humanml3d_272 import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')


##### ---- Accelerator Setup ---- #####
accelerator = Accelerator()
comp_device = accelerator.device
def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')



##### ---- Dataloader ---- #####
train_loader = dataset_tae.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

val_loader = dataset_eval_tae.DATALoader(args.dataname, False,
                                        32,
                                        unit_length=2**args.down_t)

##### ---- Network ---- #####
clip_range = [-30,20]

net = tae.Causal_HumanTAE(
                       hidden_size=args.hidden_size,
                       down_t=args.down_t,
                       stride_t=args.stride_t,
                       depth=args.depth,
                       dilation_growth_rate=args.dilation_growth_rate,
                       activation='relu',
                       latent_dim=args.latent_dim,
                       clip_range=clip_range
                       )


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt, strict=True)
net.train()
net.to(comp_device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

net, optimizer, train_loader, val_loader = accelerator.prepare(net, optimizer, train_loader, val_loader)
train_loader_iter = dataset_tae.cycle(train_loader)

Loss = losses.ReConsLoss(motion_dim=272)

##### ------ warm-up ------- #####
avg_recons, avg_kl, avg_root = 0., 0., 0.
for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float()

    if args.num_gpus > 1:
        pred_motion, mu, logvar = net.module(gt_motion)
    else:
        pred_motion, mu, logvar = net(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)

    loss_kl = Loss.forward_KL(mu, logvar)
    loss_root = Loss.forward_root(pred_motion, gt_motion)
    loss = loss_motion + loss_kl + args.root_loss * loss_root

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_kl += loss_kl.item()
    avg_root += loss_root.item()
    
    if nb_iter % args.print_iter ==  0 :
        if accelerator.is_main_process:
            avg_recons /= args.print_iter
            avg_kl /= args.print_iter
            avg_root /= args.print_iter

            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Recons.  {avg_recons:.5f} \t KL. {avg_kl:.5f} \t Root. {avg_root:.5f}")
            
        
        avg_recons, avg_kl, avg_root = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_kl, avg_root = 0., 0., 0.

if args.num_gpus > 1:
    best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(args.out_dir, val_loader, net.module, logger, writer, 0, best_iter=0, best_mpjpe=1000, device=comp_device, accelerator=accelerator)
else:
    best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(args.out_dir, val_loader, net, logger, writer, 0, best_iter=0, best_mpjpe=1000, device=comp_device, accelerator=accelerator)

for nb_iter in range(1, args.total_iter + 1):
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float() 
    
    if args.num_gpus > 1:
        pred_motion, mu, logvar = net.module(gt_motion)
    else:
        pred_motion, mu, logvar = net(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)

    loss_kl = Loss.forward_KL(mu, logvar)
 
    loss_root = Loss.forward_root(pred_motion, gt_motion)
    loss = loss_motion + loss_kl + args.root_loss * loss_root
    
    
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    
    try:
        avg_recons += loss_motion.item()
        avg_kl += loss_kl.item()
        avg_root += loss_root.item()
    except:
        continue
    
    if nb_iter % args.print_iter ==  0 :
        if accelerator.is_main_process:
            avg_recons /= args.print_iter
            avg_kl /= args.print_iter
            avg_root /= args.print_iter
            writer.add_scalar('./Train/Recon_loss', avg_recons, nb_iter)
            writer.add_scalar('./Train/KL', avg_kl, nb_iter)
            writer.add_scalar('./Train/Root_loss', avg_root, nb_iter)
            writer.add_scalar('./Train/LR', current_lr, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Recons.  {avg_recons:.5f} \t KL. {avg_kl:.5f} \t Root. {avg_root:.5f}")
        
        avg_recons, avg_kl, avg_root = 0., 0., 0.

    if nb_iter % args.eval_iter==0:
        if args.num_gpus > 1:
            best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(args.out_dir, val_loader, net.module, logger, writer, nb_iter, best_iter, best_mpjpe, device=comp_device, accelerator=accelerator)
        else:
            best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(args.out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_mpjpe, device=comp_device, accelerator=accelerator)

accelerator.wait_for_everyone()