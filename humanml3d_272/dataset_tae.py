import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class MotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 300
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            split_file = pjoin(self.data_root, 'split', 'train.txt')
            
        elif dataset_name == 't2m_babel_272':
            self.hml_data_root = './humanml3d_272'
            self.hml_motion_dir = pjoin(self.hml_data_root, 'motion_data')
            hml_split_file = pjoin(self.hml_data_root, 'split', 'train.txt')
            self.joints_num = 22
            self.max_motion_length = 300

            self.babel_data_root = './babel_272'
            self.babel_motion_dir = pjoin(self.babel_data_root, 'motion_data')
            babel_split_file = pjoin(self.babel_data_root, 'split', 'train.txt')
            self.meta_dir = pjoin(self.babel_data_root, 't2m_babel_mean_std')
        else:
            raise ValueError(f'Dataset {dataset_name} not found')

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))
        
        self.data = []
        self.lengths = []
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

        for name in tqdm(id_list):
            try:
                if dataset_name == 't2m_272':
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                elif dataset_name == 't2m_babel_272':
                    if name.split('_')[0] == 'b':
                        # Babel-272
                        motion = np.load(pjoin(self.babel_motion_dir, name.split('_')[1] + '.npy'))
                    else:
                        # HumanML3D-272
                        motion = np.load(pjoin(self.hml_motion_dir, name + '.npy'))
                else:
                    raise ValueError(f'Dataset {dataset_name} not found')

                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                pass
        
        print(f'Training on {len(self.data)} motion sequences...')
            
        self.mean = mean
        self.std = std
        

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        # Motion Normalization
        motion = (motion - self.mean) / self.std

        return motion

def DATALoader(dataset_name,
               batch_size,
               num_workers = 64,
               window_size = 64,
               unit_length = 4):
    
    trainSet = MotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x