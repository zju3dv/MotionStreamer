import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='options',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataname', type=str, default='t2m_272', help='dataset directory') 
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    return parser.parse_args()