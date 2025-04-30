import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import json


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)



def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {split_file.split('/')[-2]} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        miss_count = 0
        new_name_list = []
        length_list = []

        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                
                motion = np.load(pjoin(motion_dir, name + ".npy"))

                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format in ['vector_263', '']:
                    pass
                else:
                    raise NotImplementedError

                
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 30):int(to_tag * 30)]
                                
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                            

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
                    
            except:
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)


    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx].split('_')[-1]

        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        # caption, tokens = text_data["caption"], text_data["tokens"]
        caption = text_data["caption"]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Normalization"
        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            caption,
            motion,
            m_length,
            retrieval_name
        )
