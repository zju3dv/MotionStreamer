import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 4, latent_dir=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        
        if dataset_name == 't2m_babel_272':
            # Babel-272-stream data dir
            self.babel_stream_data_root = './babel_272_stream' 
            self.babel_stream_text_dir = pjoin(self.babel_stream_data_root, 'train_stream_text')
            fps = 30
            self.max_motion_length = 78

            # HumanML3D-272 data dir
            self.hml_data_root = './humanml3d_272'
            self.hml_text_dir = pjoin(self.hml_data_root, 'texts')
    
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}')
        
        id_list = []
        
        for file in os.listdir(latent_dir):
            if file.endswith('.npy'):
                id_list.append(file[:-4])   
                
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            m_token_list = np.load(pjoin(latent_dir, '%s.npy'%name))

            if len(m_token_list) > self.max_motion_length:
                continue

            # Read text
            if name.split('_')[0] == 'seq':
                # Babel-272-stream
                with cs.open(pjoin(self.babel_stream_text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        text_dict = {}
                        B_split = line.strip().split('*')[1].split('#')
                        B_text = line.strip().split('*')[1].split('#')[0]
                        if B_text == '':
                            continue
                        B_t_tokens = B_split[1].split(' ')
                        A_motion_length = B_split[-1]
                        A_token_length = int(A_motion_length) // unit_length
                        text_dict['caption'] = B_text   
                        text_dict['tokens'] = B_t_tokens
                        
                        flag = True                                
                        text_data.append(text_dict)

            else:
                # HumanML3D-272
                with cs.open(pjoin(self.hml_text_dir, name + '.txt')) as f:
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

                        A_token_length = 0

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
                                                        'text':[text_dict],
                                                        'A_token_length': A_token_length
                                                        }
                            new_name_list.append(new_name)

            if flag:
                
                data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data,
                                    'A_token_length': A_token_length
                                    }
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


        A_token_length = data['A_token_length']
        m_tokens_len = m_tokens.shape[0]

        
        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.zeros((self.max_motion_length - m_tokens_len, m_tokens.shape[1]), dtype=int)], axis=0)


        return caption, m_tokens, m_tokens_len, A_token_length




def DATALoader(dataset_name,
                batch_size, unit_length=4,
                num_workers = 8, latent_dir = None) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, unit_length=unit_length, latent_dir=latent_dir),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

