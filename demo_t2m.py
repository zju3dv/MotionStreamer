import os
import torch
import numpy as np
from models.llama_model import LLaMAHF, LLaMAHFConfig
import models.tae as tae
import options.option_transformer as option_trans
import warnings
warnings.filterwarnings('ignore')

comp_device = torch.device('cuda')
##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

from sentence_transformers import SentenceTransformer
t5_model = SentenceTransformer('sentencet5-xxl/')
t5_model.eval()
for p in t5_model.parameters():
    p.requires_grad = False

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


reference_end_latent = np.load('reference_end_latent_t2m_272.npy')
reference_end_latent = torch.from_numpy(reference_end_latent).to(comp_device)

mean = np.load('humanml3d_272/mean_std/Mean.npy')
std = np.load('humanml3d_272/mean_std/Std.npy')

# forward inference
threshold = 0.1
motion_latents = trans_encoder.sample_for_eval_CFG_inference(text=args.text, tokenizer=t5_model, device=comp_device, reference_end_latent=reference_end_latent, threshold=threshold)

# forward decode
motion_seqs = net.forward_decoder(motion_latents)
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d

motion = motion_seqs.squeeze(0)
motion = motion.detach().cpu().numpy()

if not os.path.exists('demo_output'):
    os.makedirs('demo_output')

if args.mode == 'pos':
    # Option1: recover from joint position
    pred_xyz = recover_from_local_position(motion * std + mean, 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    pose_vis = plot_3d.draw_to_batch(xyz, [args.text], [f'demo_output/{args.text}.mp4'], fps=30)
    print(f"Visualized result is saved in demo_output/{args.text}.mp4")

elif args.mode == 'rot':
    # Option2: recover from joint rotation
    # In our 272-dim representation, Inverse Kinematics (IK) is not needed.
    np.save('demo_output/global_rotation.npy', motion) 
    print("You can further convert to BVH format and visualize in Blender following: https://github.com/Li-xingXiao/272-dim-Motion-Representation?tab=readme-ov-file#6-representation_272-to-bvh-conversion-optional (Step 6: Representation_272 to BVH conversion)")
            
else:
    raise ValueError(f'Invalid mode: {args.mode}')

