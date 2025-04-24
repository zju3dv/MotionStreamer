import os 
import torch
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_tae
import options.option_transformer as option_trans
import warnings
warnings.filterwarnings('ignore')


comp_device = torch.device('cuda')
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
val_loader = dataset_eval_tae.DATALoader(args.dataname, True, 32)


# load evaluator:--------------------------------
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4)

ckpt_path = 'Evaluator_272/epoch=99.ckpt'
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

gt_fid, gt_div, gt_top1, gt_top2, gt_top3, gt_matching = eval_trans.evaluation_gt(val_loader, evaluator, device=comp_device)
    
print('final result:')
print(f'gt_fid: {gt_fid}')
print(f'gt_div: {gt_div}')
print(f'gt_top1: {gt_top1}')
print(f'gt_top2: {gt_top2}')
print(f'gt_top3: {gt_top3}')
print(f'gt_matching: {gt_matching}')


