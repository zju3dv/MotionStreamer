import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 4, latent_dir=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            fps = 30
            self.max_motion_length = 78   
            dim_pose = 272
            split_file = pjoin(self.data_root, 'split', 'train.txt')

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
     
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(latent_dir, '%s.npy'%name))
            except:
                continue

            # Read text
            with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                text_data = []
                flag = False
                lines = f.readlines()

                for line in lines:
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    t_tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length):
                            m_token_list_new = [m_token_list[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)]] 

                        if len(m_token_list_new) == 0:
                            continue

                        new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                        data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict]}
                        new_name_list.append(new_name)
                    
            if flag:
                data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data}
                new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = np.array(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']

        if len(m_tokens.shape) == 3:
            m_tokens = m_tokens.squeeze(0)
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]
        
        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.zeros((self.max_motion_length-m_tokens_len, m_tokens.shape[1]), dtype=int)], axis=0)
        return caption, m_tokens, m_tokens_len




def DATALoader(dataset_name,
                batch_size, latent_dir, unit_length=4,
                num_workers = 8) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, latent_dir = latent_dir, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

