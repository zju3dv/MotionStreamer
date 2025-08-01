import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import sys
from models.llama_model import LLaMAHF, LLaMAHFConfig
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m
import models.tae as tae
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd()) 

comp_device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
val_loader = dataset_eval_t2m.DATALoader(args.dataname, True, 32)

##### ---- Network ---- #####
from sentence_transformers import SentenceTransformer
t5_model = SentenceTransformer('../sentencet5-xxl/')
t5_model.eval()
for p in t5_model.parameters():
    p.requires_grad = False
tokenize_model = t5_model

# Causal TAE

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

config = LLaMAHFConfig.from_name('Normal_size')
config.block_size = 78
trans_encoder = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(comp_device)


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
trans_encoder.eval()
trans_encoder.to(comp_device)

# load evaluator:
import torch
from transformers import AutoTokenizer, AutoModel
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict

modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = 'epoch=99.ckpt'
print(f'Loading evaluator checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path)
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
div = []
top1 = []
top2 = []
top3 = []
matching = []
mpjpe = []

best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_trans.evaluation_transformer_272_single(val_loader, net, trans_encoder, tokenize_model, logger, evaluator, 4.0)
fid.append(best_fid)
div.append(best_div)
top1.append(best_top1)
top2.append(best_top2)
top3.append(best_top3)
matching.append(best_matching)

logger.info('final result:')
logger.info(f'fid: {fid}')
logger.info(f'div: {div}')
logger.info(f'top1: {top1}')
logger.info(f'top2: {top2}')
logger.info(f'top3: {top3}')
logger.info(f'MM-dist (matching score) : {matching}')
