import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    return default_collate(batch)


class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, max_text_len = 20, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 300
            fps = 30
            self.meta_dir = './humanml3d_272/mean_std'
            if is_test:
                split_file = pjoin(self.data_root, 'split', 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'split', 'val.txt')
        elif dataset_name == 't2m_babel_272':
            # HumanML3D-272 data
            self.hml_data_root = './humanml3d_272'
            self.hml_motion_dir = pjoin(self.hml_data_root, 'motion_data')
            if is_test:
                hml_split_file = pjoin(self.hml_data_root, 'split', 'test.txt')
            else:
                hml_split_file = pjoin(self.hml_data_root, 'split', 'val.txt')

            self.joints_num = 22
            self.max_motion_length = 300
            fps = 30

            # Babel-272 data
            self.babel_data_root = './babel_272'
            self.babel_motion_dir = pjoin(self.babel_data_root, 'motion_data')
            babel_split_file = pjoin(self.babel_data_root, 'split', 'val.txt')
            self.meta_dir = pjoin(self.babel_data_root, 't2m_babel_mean_std')

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy')) 
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        min_motion_len = 60  # 30 fps

        data_dict = {}
        id_list = []

        if dataset_name == 't2m_272':
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

        elif dataset_name == 't2m_babel_272':
            with cs.open(hml_split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
            with cs.open(babel_split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append('b_' + line.strip())

        new_name_list = []
        length_list = []


        for name in tqdm(id_list):
            try:
                if dataset_name == 't2m_272':
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                elif dataset_name == 't2m_babel_272':
                    if name.split('_')[0] == 'b':
                        motion = np.load(pjoin(self.babel_motion_dir, name.split('_')[1] + '.npy'))
                    else:
                        motion = np.load(pjoin(self.hml_motion_dir, name + '.npy'))

                if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                    continue

                data_dict[name] = {'motion': motion,
                                    'length': len(motion)
                                        }
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']


        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        #"Motion Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return motion, m_length




def DATALoader(dataset_name, is_test,
                batch_size,
                num_workers = 64, unit_length = 4, drop_last=True) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, unit_length=unit_length),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = drop_last)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x