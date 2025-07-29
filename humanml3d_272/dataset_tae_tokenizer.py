import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os


class MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40
        
        
        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            split_file = pjoin(self.data_root, 'split', 'train.txt')

        elif dataset_name == 't2m_babel_272':
            # HumanML3D-272 data dir
            self.hml_data_root = './humanml3d_272'
            self.hml_motion_dir = pjoin(self.hml_data_root, 'motion_data')
            hml_split_file = pjoin(self.hml_data_root, 'split', 'train.txt')

            # Babel-272-stream data dir
            self.babel_stream_data_root = './babel_272_stream'
            self.babel_stream_motion_dir = pjoin(self.babel_stream_data_root, 'train_stream')
            self.meta_dir = './babel_272/t2m_babel_mean_std'

        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
            
    
        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))
        
        data_dict = {}
        id_list = []

        if dataset_name == 't2m_272':
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
        elif dataset_name == 't2m_babel_272':
            # HumanML3D-272 data
            with cs.open(hml_split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            # Babel-272-stream data
            for file in os.listdir(self.babel_stream_motion_dir):
                if file.endswith('.npy'):
                    id_list.append(file[:-4])   # seq_1, seq_2, ...

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                if dataset_name == 't2m_272':
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len:
                        continue
                elif dataset_name == 't2m_babel_272':
                    if name.split('_')[0] == 'seq':
                        # seq_1, seq_2, ... (Babel-272-stream)
                        motion = np.load(pjoin(self.babel_stream_motion_dir, name + '.npy'))
                    else:
                        # (HumanML3D-272)
                        motion = np.load(pjoin(self.hml_motion_dir, name + '.npy'))
                        if (len(motion)) < min_motion_len:
                            continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        # "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4) : 
    
    train_loader = torch.utils.data.DataLoader(MotionDataset(dataset_name, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x