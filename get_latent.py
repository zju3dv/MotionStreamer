import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json
import models.tae as tae
import options.option_tae as option_tae
import utils.utils_model as utils_model
from humanml3d_272 import dataset_tae_tokenizer
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader = dataset_tae_tokenizer.DATALoader(args.dataname)

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

logger.info('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()


##### ---- get reference end latent ---- #####
reference_end_pose = torch.zeros(1, 4, 272).cuda()   # impossible pose prior
reference_end_latent, _, _ = net.encode(reference_end_pose)
reference_end_latent = reference_end_latent.permute(1,0)
np.save(f'reference_end_latent_{args.dataname}.npy', reference_end_latent.cpu().detach().numpy())

os.makedirs(args.latent_dir, exist_ok = True)

for batch in tqdm(train_loader):
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]
    pose = pose.cuda().float()
    latent, _, _  = net.encode(pose)
    latent = latent.permute(1,0)
    latent = torch.cat([latent, reference_end_latent], dim=0)
    latent = latent.cpu().detach().numpy()
    np.save(pjoin(args.latent_dir, name[0] +'.npy'), latent)
