import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='options',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataname', type=str, default='t2m_272', help='dataset directory') 
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for training. ')
    parser.add_argument('--latent_dir', type=str, default='latent/', help='latent directory')
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for causal TAE')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument('--out-dir', type=str, default='output_GPT_Final/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--hidden_size', default=1024, type=int, help='hidden size')
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")

    parser.add_argument('--num_diffusion_head_layers', type=int, default=9, help='number of diffusion head layers')
    parser.add_argument('--latent_dim', type=int, default=16, help='latent dimension')
    parser.add_argument('--total_iter', type=int, default=100000, help='total iteration')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--decay-option',default='all', type=str, choices=['all'], help='weight decay option')
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='optimizer')

    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs')
    parser.add_argument('--total-iter', default=2000000, type=int, help='number of total iterations to run')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')

    parser.add_argument('--text', type=str, default='A man is jogging around.')
    parser.add_argument('--mode', type=str, default='pos', choices=['pos', 'rot'], help='recover mode, pos: position, rot: rotation')



    return parser.parse_args()