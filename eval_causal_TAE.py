import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import models.tae as tae
import options.option_tae as option_tae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_tae
import sys
import warnings
warnings.filterwarnings('ignore')

os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd()) 


comp_device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

val_loader = dataset_eval_tae.DATALoader(args.dataname, True, 32)

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


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(comp_device)


# load evaluator:--------------------------------
import torch
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, latent_dim=256, max_len=300)

ckpt = torch.load('epoch=99.ckpt')

# load textencoder
textencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "textencoder":
        name = k.replace("textencoder.", "")
        textencoder_ckpt[name] = v
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(comp_device)

# load motionencoder
motionencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "motionencoder":
        name = k.replace("motionencoder.", "")
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)
#--------------------------------

evaluator = [textencoder, motionencoder]

fid = []
mpjpe = []

best_fid, best_mpjpe, writer, logger = eval_trans.evaluation_tae_single(args.out_dir, val_loader, net, logger, writer, evaluator=evaluator, device=comp_device)
fid.append(best_fid)
mpjpe.append(best_mpjpe)

logger.info('final result:')
logger.info(f'fid: {fid}')
logger.info(f'mpjpe: {mpjpe} (mm)')