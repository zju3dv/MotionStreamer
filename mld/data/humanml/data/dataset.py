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

from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer
import json
import time

# import spacy
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

"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
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
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
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
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


"""For use of training text motion matching model, and evaluations"""


# class Text2MotionDatasetV2(data.Dataset):

#     def __init__( 
#         self,
#         mean,
#         std,
#         split_file,
#         w_vectorizer,
#         max_motion_length,
#         min_motion_length,
#         max_text_len,
#         unit_length,
#         motion_dir,
#         text_dir,
#         input_format, 
#         njoints, 
#         tiny=False,
#         debug=False,
#         progress_bar=True,
#         **kwargs,
#     ):
#         import pdb; pdb.set_trace()
#         self.w_vectorizer = w_vectorizer
#         self.max_length = 20
#         self.pointer = 0
#         self.max_motion_length = max_motion_length
#         # min_motion_len = 40 if dataset_name =='t2m' else 24

#         self.min_motion_length = min_motion_length
#         self.max_text_len = max_text_len
#         self.unit_length = unit_length

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         self.id_list = id_list

#         if tiny or debug:
#             progress_bar = False
#             maxdata = 10 if tiny else 100
#         else:
#             maxdata = 1e10
#         # import pdb; pdb.set_trace()

#         if progress_bar:
#             enumerator = enumerate(
#                 track(
#                     id_list,
#                     f"Loading {split_file.split('/')[-2]} {split_file.split('/')[-1].split('.')[0]}",
#                 ))
#         else:
#             enumerator = enumerate(id_list)
#         count = 0
#         bad_count = 0
#         miss_count = 0
#         new_name_list = []
#         length_list = []
#         # import pdb; pdb.set_trace()
#         for i, name in enumerator:
#             if count > maxdata:
#                 break
#             try:
#                 #     import pdb; pdb.set_trace()
#                 motion = np.load(pjoin(motion_dir, name + ".npy"))

#                 # import pdb; pdb.set_trace()
#                 if input_format == 'root_position':
#                     motion = motion[..., :4+(njoints-1)*3]
#                 elif input_format == 'root_position_vel':
#                     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
#                     # import pdb; pdb.set_trace()
#                 elif input_format == 'root_position_rot6d':
#                     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
#                 elif input_format == 'root_rot6d':
#                     motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
#                 elif input_format == 'vector_263':
#                     pass
#                 else:
#                     print('NotImplementedError')
#                     raise NotImplementedError

#                 if (len(motion)) < self.min_motion_length or (len(motion) >=
#                                                                 300):
#                     print(len(motion))
#                     bad_count += 1
#                     continue
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(text_dir, name + ".txt")) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split("#")
#                         caption = line_split[0]
#                         try:
#                             tokens = line_split[1].split(" ")
#                         except:
#                             import pdb; pdb.set_trace()
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict["caption"] = caption
#                         text_dict["tokens"] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 n_motion = motion[int(f_tag * 30):int(to_tag *
#                                                                         30)]
#                                 if (len(n_motion)
#                                     ) < self.min_motion_length or (
#                                         (len(n_motion) >= 300)):
                                    
#                                     continue
#                                 new_name = (
#                                     random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
#                                     "_" + name)
#                                 while new_name in data_dict:
#                                     new_name = (random.choice(
#                                         "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
#                                                 name)
#                                 data_dict[new_name] = {
#                                     "motion": n_motion,
#                                     "length": len(n_motion),
#                                     "text": [text_dict],
#                                 }
#                                 new_name_list.append(new_name)
#                                 length_list.append(len(n_motion))
#                             except:
#                                 # None
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag,
#                                         to_tag, name)
#                                 # break

#                 if flag:
#                     data_dict[name] = {
#                         "motion": motion,
#                         "length": len(motion),
#                         "text": text_data,
#                     }
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#                     # print(count)
#                     count += 1
#                     # print(name)
#             except:
#                 # import pdb; pdb.set_trace()
#                 miss_count += 1
#                 pass

#         print(f'Here are {miss_count} not in dataset!')
#         print(f'Here are {bad_count} either small or large than given value.')

#         name_list, length_list = zip(
#             *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
#         self.mean = mean
#         self.std = std

#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.nfeats = motion.shape[1]
#         self.name_list = name_list
#         self.reset_max_len(self.max_length)
#         # train 24546
#         # test 4648
#         print('train len', len(data_dict))
#         print('test len', len(data_dict))
#         import pdb; pdb.set_trace()



#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.name_list) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]

#         retrieval_name = self.name_list[idx].split('_')[-1]

#         motion, m_length, text_list = data["motion"], data["length"], data[
#             "text"]

#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption, tokens = text_data["caption"], text_data["tokens"]

#         if len(tokens) < self.max_text_len:
#             # pad with "unk"
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#             tokens = tokens + ["unk/OTHER"
#                                ] * (self.max_text_len + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.max_text_len]
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         # Crop the motions in to times of 4, and introduce small variations
#         if self.unit_length < 10:
#             coin2 = np.random.choice(["single", "single", "double"])
#         else:
#             coin2 = "single"

#         if coin2 == "double":
#             m_length = (m_length // self.unit_length - 1) * self.unit_length
#         elif coin2 == "single":
#             m_length = (m_length // self.unit_length) * self.unit_length
#         idx = random.randint(0, len(motion) - m_length)
#         motion = motion[idx:idx + m_length]
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         # # padding
#         # if m_length < self.max_motion_length:
#         #     motion = np.concatenate(
#         #         [
#         #             motion,
#         #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
#         #         ],
#         #         axis=0,
#         #     )
#         # print(word_embeddings.shape, motion.shape, m_length)
#         # print(tokens)

#         # debug check nan
#         if np.any(np.isnan(motion)):
#             raise ValueError("nan in motion")

#         return (
#             word_embeddings,
#             pos_one_hots,
#             caption,
#             sent_len,
#             motion,
#             m_length,
#             "_".join(tokens),
#             retrieval_name
#         )
#         # return caption, motion, m_length





class Text2MotionDatasetV2_VQToken(data.Dataset):

    def __init__(
        self,
        cfg, 
        mean,
        std,
        split_file,
        w_vectorizer,
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
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        
        # import pdb; pdb.set_trace()
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        self.max_motion_token_length = max_motion_length // unit_length + 2

        # import pdb; pdb.set_trace()
        self.codebook_size = cfg.model.motion_vae.params.nb_code
        self.mot_end_idx = self.codebook_size
        self.mot_pad_idx = self.codebook_size + 1

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
        # import pdb; pdb.set_trace()

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
        
        odd_token_count = 0

        miss_VQ_count = 0

        new_name_list = []
        length_list = []

        token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/humanml3d/vq_tokens/{cfg.model.vae_type}')
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break


            try:
                #     import pdb; pdb.set_trace()

                motion = np.load(pjoin(motion_dir, name + ".npy"))

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'vector_263':
                    pass
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                200):
                    bad_count += 1
                    continue

                
                motion_token_path = pjoin(f'/comp_robot/lushunlin/datasets/humanml3d/vq_tokens/{cfg.model.vae_type}', name + ".npy")
                if motion_token_path not in token_all_path:
                    miss_VQ_count += 1
                    continue

                motion_token = np.load(motion_token_path)

                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
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
                            # import pdb; pdb.set_trace()
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                if int(f_tag*20/unit_length) < int(to_tag*20/unit_length):
                                    motion_token_new = [m_tokens[int(f_tag*20/unit_length) : int(to_tag*20/unit_length)] for m_tokens in motion_token]
                                else:
                                    motion_token_new = []

                                if len(motion_token_new) == 0:
                                    continue

                                if int(to_tag*20/2) - int(f_tag*20/2) != 2 * (int(to_tag*20/4) - int(f_tag*20/4)):
                                    odd_token_count += 1
                                    continue

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
                                    "motion_vq_token": motion_token_new
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # import pdb; pdb.set_trace()
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "motion_vq_token": motion_token
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass
            
            # import pdb; pdb.set_trace()


        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        print(f'Here are {miss_VQ_count} not in VQ')
        print(f'Here are {odd_token_count} odd token_count!')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))
        # import pdb; pdb.set_trace()



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

        retrieval_name = self.name_list[idx]



        motion, m_length, text_list, motion_token_list = data["motion"], data["length"], data[
            "text"], data['motion_vq_token']

        # Randomly select a caption
        text_data = random.choice(text_list)
        motion_token = random.choice(motion_token_list)

        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

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
        "Z Normalization"
        motion = (motion - self.mean) / self.std


        m_tokens_len = motion_token.shape[0]

        if m_tokens_len+1 < self.max_motion_token_length:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_token_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # if m_tokens_len+1 < self.max_motion_length:
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        # else:
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            retrieval_name,
            m_tokens, 
            m_tokens_len
        )




class Text2MotionDatasetV2_Dual_codebook_VQToken(data.Dataset):

    def __init__(
        self,
        cfg, 
        mean,
        std,
        split_file,
        w_vectorizer,
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
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        
        # import pdb; pdb.set_trace()
        assert cfg.model.vae_type in ['hvq']
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.unit_length_b = 2
        self.unit_length_t = 4

        self.max_motion_token_length = 98 + 49 + 2

        # import pdb; pdb.set_trace()
        self.codebook_size = cfg.model.motion_vae.params.nb_code
        self.mot_end_idx = self.codebook_size
        self.mot_pad_idx = self.codebook_size + 1

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
        # import pdb; pdb.set_trace()

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

        miss_VQ_count = 0

        odd_token_count = 0

        new_name_list = []
        length_list = []

        token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/humanml3d/vq_tokens/{cfg.model.vae_type}')
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break


            try:
                #     import pdb; pdb.set_trace()

                motion = np.load(pjoin(motion_dir, name + ".npy"))

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'vector_263':
                    pass
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                200):
                    bad_count += 1
                    continue

                motion_token_path_codebook1 = pjoin(f'/comp_robot/lushunlin/datasets/humanml3d/vq_tokens/{cfg.model.vae_type}/motion_vqcode_b', name + ".npy")
                motion_token_path_codebook2 = pjoin(f'/comp_robot/lushunlin/datasets/humanml3d/vq_tokens/{cfg.model.vae_type}/motion_vqcode_t', name + ".npy")

                if motion_token_path_codebook1 not in token_all_path or motion_token_path_codebook2 not in token_all_path:
                    miss_VQ_count += 1
                    continue

                motion_token_codebook1 = np.load(motion_token_path_codebook1)
                motion_token_codebook2 = np.load(motion_token_path_codebook2)

                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
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
                            # import pdb; pdb.set_trace()
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                # if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b):
                                #     motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1]
                                # else:
                                #     motion_token_new_codebook1 = []

                                motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1 if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b)]
                                    

                                # if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t):
                                #     motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2]
                    # else:
                                #     motion_token_new_codebook2 = []            

                                motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2 if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t)]

                                if len(motion_token_new_codebook1) == 0 or len(motion_token_new_codebook2) == 0:
                                    continue

                                # if motion_token_new_codebook1[0].shape[0] % 2 != 0:
                                #     odd_token_count += 1
                                #     motion_token_new_codebook1 = [motion_token_new_codebook1[0][:-1]]
                                    # continue
                                if int(to_tag*20/self.unit_length_b) - int(f_tag*20/self.unit_length_b) != 2 * (int(to_tag*20/self.unit_length_t) - int(f_tag*20/self.unit_length_t)):
                                    odd_token_count += 1
                                    continue

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
                                    "motion_vq_token_codebook1": motion_token_new_codebook1, 
                                    "motion_vq_token_codebook2": motion_token_new_codebook2 
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                import pdb; pdb.set_trace()
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "motion_vq_token_codebook1": motion_token_codebook1, 
                        "motion_vq_token_codebook2": motion_token_codebook2 
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass
            
            # import pdb; pdb.set_trace()


        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        print(f'Here are {miss_VQ_count} not in VQ')

        print(f'Hare are {odd_token_count} odd codebook1_tokens!')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))

        # list_1 = [1,2,3,4,5,6]
        # list_2 = [1,2,3]

        # return_list = self.compose_new_list(list_1, list_2)
        for name in name_list:
            data_codebook1 = random.choice(data_dict[name]['motion_vq_token_codebook1'])
            data_codebook2 = random.choice(data_dict[name]['motion_vq_token_codebook2'])

            try:
                if data_codebook1.shape[0] != 2 * data_codebook2.shape[0]:
                    print('***************************')
                    print(name)
                    import pdb; pdb.set_trace()
            except:
                print('error')
                import pdb; pdb.set_trace()
    
        print('load done')
        # import pdb; pdb.set_trace()



    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compose_new_list(self, list_1, list_2):
        # print('******************')
        # print('list1: {}, list2: {}'.format(len(list_1), len(list_2)))
        
        assert len(list_1) == 2 * len(list_2)

        new_list = []

        for i in range(len(list_1)):
            new_list.append(list_1[i])
            if i % 2 == 1:
                new_list.append(list_2[i // 2])

        return new_list

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]



        motion, m_length, text_list, motion_token_list_codebook1, motion_token_list_codebook2 = data["motion"], data["length"], data[
            "text"], data['motion_vq_token_codebook1'], data['motion_vq_token_codebook2']

        # m_token_list = []
        # for i in range(len(motion_token_list_codebook2)):
        #     m_token_list_i = [x for pair in zip(motion_token_list_codebook1[i], motion_token_list_codebook2[i]) for x in pair]
        #     m_token_list.append(m_token_list_i)

        # m_token_list = np.array(m_token_list)

        m_token_list = []
        for i in range(len(motion_token_list_codebook1)):
            m_token_list_i = self.compose_new_list(motion_token_list_codebook1[i].tolist(), motion_token_list_codebook2[i].tolist())
            m_token_list.append(m_token_list_i)


        m_token_list = np.array(m_token_list)

        # import pdb; pdb.set_trace()
        # Randomly select a caption
        text_data = random.choice(text_list)
        motion_token = random.choice(m_token_list)

        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

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
        "Z Normalization"
        motion = (motion - self.mean) / self.std


        m_tokens_len = motion_token.shape[0]

        if m_tokens_len+1 < self.max_motion_token_length:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_token_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # if m_tokens_len+1 < self.max_motion_length:
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        # else:
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            retrieval_name,
            m_tokens, 
            m_tokens_len
        )


"""For use of training baseline"""


class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
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
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
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
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

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
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        # padding
        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class VQMotionDataset(data.Dataset):
    def __init__(self, mean, std, split_file, max_motion_length, min_motion_length, unit_length, motion_dir, input_format, njoints, tiny=False, debug=False, progress_bar=True, **kwargs, ):

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length # 196
        self.min_motion_length = min_motion_length # 40
        self.unit_length = unit_length

        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                #     import pdb; pdb.set_trace()
                motion = np.load(pjoin(motion_dir, name + ".npy"))

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'vector_263':
                    pass
                elif input_format == 'root_body_pos_vel_hand_pos_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or len(motion) >= 200:
                    continue

                data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "name": name,
                }

                new_name_list.append(name)
                length_list.append(len(motion))
                # print(count)
                count += 1
                # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))
        import pdb; pdb.set_trace()

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name, m_length




class VQMotionDataset_MotionX(data.Dataset):
    def __init__(self, mean, std, split_file, max_motion_length, min_motion_length, unit_length, motion_dir, motion_type, input_format, njoints, tiny=False, debug=False, progress_bar=True, **kwargs, ):

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length # 196
        self.min_motion_length = min_motion_length # 40
        self.unit_length = unit_length

        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()

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
        miss_humanml_count = 0
        new_name_list = []
        length_list = []
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                #     import pdb; pdb.set_trace()
                motion = np.load(pjoin(motion_dir, motion_type, name + ".npy"))

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format in ['vector_263', 'smplx_212', 'smplx_159']:
                    pass
                elif input_format == 'root_body_pos_vel_hand_pos_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if 'humanml' in name:
                    if (len(motion)) < self.min_motion_length or len(motion) >= 200:
                        miss_humanml_count += 1
                        continue
                else:
                    if (len(motion)) < self.min_motion_length:
                        bad_count += 1
                        continue
                    elif len(motion) >= self.max_motion_length:
                        start = random.randint(0,len(motion) - self.max_motion_length)
                        motion = motion[start:start+self.max_motion_length]

                data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "name": name,
                }

                new_name_list.append(name)
                length_list.append(len(motion))
                # print(count)
                count += 1
                # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')
        print(f'Here are {miss_humanml_count} miss in humanml')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))


        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))
        # import pdb; pdb.set_trace()

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name, m_length

class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
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
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        # import pdb; pdb.set_trace()
        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        # import pdb; pdb.set_trace()
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                #     import pdb; pdb.set_trace()
                motion = np.load(pjoin(motion_dir, name + ".npy"))

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format in ['vector_263', '']:
                    pass
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                400):
                    # print(len(motion))
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
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
                                n_motion = motion[int(f_tag * 30):int(to_tag *
                                                                        30)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 400)):
                                    
                                    continue
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
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))
        # import pdb; pdb.set_trace()



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

        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

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
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            retrieval_name
        )
        # return caption, motion, m_length


"""For use of training baseline"""


class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
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
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
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
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

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
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        # padding
        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
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
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):

    def __init__(self,
                 mode,
                 datapath="./dataset/kit_opt.txt",
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)


def get_njoints(dataset_name):
    if dataset_name == 'humanml3d':
        njoints = 22
    elif dataset_name == 'kit':
        njoints = 21
    elif dataset_name in ['motionx', 'motionx_v25', 'motionx_v26']:
        njoints = 52
    else:
        raise NotImplementedError
    
    return njoints

'''For Motion-X'''
class Text2MotionDatasetMotionX(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type,
        input_format, 
        hand_mask,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        # import pdb; pdb.set_trace() 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length
        self.input_format = input_format
        self.hand_mask = hand_mask

        njoints = get_njoints(dataset_name)

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name in ['motionx', 'motionx_v25', 'motionx_v26']:
            min_motion_len = 40 
        else:
            min_motion_len = 24

        # 



        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        # import pdb; pdb.set_trace()
        count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            
            # try:
            motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))

            # import pdb; pdb.set_trace()
            if input_format == 'root_position':
                motion = motion[..., :4+(njoints-1)*3]
            elif input_format == 'root_position_vel':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # import pdb; pdb.set_trace()
            elif input_format == 'root_position_rot6d':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
            elif input_format == 'root_rot6d':
                motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
            elif input_format in ['all', 'smplx_212', 'smplx_159']:
                pass
            elif input_format == 'root_body_pos_vel_hand_all':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos_vel':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_rot':
                motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
            elif input_format == 'root_position_vel_only_body':
                motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
                body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
                left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1) # 45
                right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1) # 45

                body_vel_motion = motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
                left_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1)
                right_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1)

                motion = np.concatenate((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)
            elif input_format == 'vector_263_ori_humanml':
                pass
            else:
                print('NotImplementedError')
                raise NotImplementedError
            
            if (len(motion)) < min_motion_len:
                continue
            elif len(motion) >= self.max_motion_length:
                start = random.randint(0,len(motion) - self.max_motion_length)
                motion = motion[start:start+self.max_motion_length]
            text_data = []
            flag = False
            with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    if 'humanml' in name:
                        if text_source == 'token':
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        elif text_source in ['only_text_token',  'caption']:
                            
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        else:
                            raise NotImplementedError


                    else:
                        text_dict = {}
                        line_split = line.strip()
                        caption = line_split
                        tokens = caption.split(' ')
                        f_tag = 0.0 
                        to_tag = 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)



            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            # except:
            #     import pdb; pdb.set_trace()
            #     pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        # import pdb; pdb.set_trace()
        
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]

        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']



        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
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

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)

        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        joint_mask = np.ones_like(motion)

        if self.hand_mask:
            if self.input_format == "root_body_pos_vel_hand_pos_vel":
                
                if 'humanml' in retrieval_name or 'EgoBody' in retrieval_name:
                    joint_mask[:, 4 + 21 * 3: 4 + 51 * 3] = 0
                    joint_mask[:, 4+51*3+22*3:] = 0
            else:
                raise NotImplementedError

        assert joint_mask.shape == motion.shape

        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name, joint_mask




class Text2MotionDatasetMotionX_with_face(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type,
        input_format, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        # import pdb; pdb.set_trace() 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length
        self.input_format = input_format

        njoints = get_njoints(dataset_name)

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name in ['motionx', 'motionx_v25', 'motionx_v26']:
            min_motion_len = 40 
        else:
            min_motion_len = 24

        # 



        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        # import pdb; pdb.set_trace()
        count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            
            try:
                motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))
                face_motion = np.load(pjoin(motion_dir, 'face_jaw_data', name + '.npy'))

                if face_motion.shape[0] >= motion.shape[0]:
                    face_motion = face_motion[:motion.shape[0]]
                else:
                    face_motion = torch.from_numpy(face_motion)
                    face_motion = face_motion[None].permute(0, 2, 1) # [1, num_feats, num_frames]
                    face_motion = F.interpolate(face_motion, size=motion.shape[0], mode='linear')   # motion interpolate
                    face_motion = face_motion.permute(0, 2, 1)[0].numpy()   # [num_frames, num_feats]

                assert motion.shape[0] == face_motion.shape[0]

                # import pdb; pdb.set_trace()
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                    # import pdb; pdb.set_trace()
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format in ['all', 'smplx_212', 'smplx_159']:
                    pass
                elif input_format == 'root_body_pos_vel_hand_all':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                elif input_format == 'root_body_pos_vel_hand_pos_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                elif input_format == 'root_body_pos_vel_hand_pos':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=-1)
                elif input_format == 'root_body_pos_vel_hand_rot':
                    motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                elif input_format == 'root_position_vel_only_body':
                    motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                elif input_format == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
                    body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
                    left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1) # 45
                    right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1) # 45

                    body_vel_motion = motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
                    left_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1)
                    right_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1)

                    motion = np.concatenate((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)
                else:
                    print('NotImplementedError')
                    raise NotImplementedError


                
                
                if (len(motion)) < min_motion_len:
                    continue
                elif len(motion) >= self.max_motion_length:
                    start = random.randint(0,len(motion) - self.max_motion_length)
                    motion = motion[start:start+self.max_motion_length]
                    face_motion = face_motion[start:start+self.max_motion_length]

                text_data = []
                flag = False
                with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        if 'humanml' in name:
                            if text_source == 'token':
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        n_face_motion = face_motion[int(f_tag*20) : int(to_tag*20)]

                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                                'face_motion': n_face_motion, 
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            elif text_source in ['only_text_token',  'caption']:
                                
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        n_face_motion = face_motion[int(f_tag*20) : int(to_tag*20)]

                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                                'face_motion': n_face_motion, 
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            else:
                                raise NotImplementedError


                        else:
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split
                            tokens = caption.split(' ')
                            f_tag = 0.0 
                            to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)



                if flag:
                    data_dict[name] = {'motion': motion,
                                        'face_motion': face_motion, 
                                        'length': len(motion),
                                        'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                import pdb; pdb.set_trace()
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std

        ####FIXME: 写死了
        self.face_mean = np.load('/comp_robot/lushunlin/datasets/Motion-X/mean_std/version1/face_jaw_data/mean.npy')
        self.face_std = np.load('/comp_robot/lushunlin/datasets/Motion-X/mean_std/version1/face_jaw_data/std.npy')
        #########

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        # import pdb; pdb.set_trace()
        
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]

        motion, m_length, text_list = data['motion'], data['length'], data['text']
        face_motion = data['face_motion']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']



        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
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
        face_motion = face_motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)
        face_motion = (face_motion - self.face_mean) / (self.face_std + 1e-7)

        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)

        if self.input_format == "root_body_pos_vel_hand_pos_vel":
            
            joint_mask = np.ones_like(motion)

            if 'humanml' in retrieval_name or 'EgoBody' in retrieval_name:
                joint_mask[:, 4 + 21 * 3: 4 + 51 * 3] = 0
                joint_mask[:, 4+51*3+22*3:] = 0
        else:
            raise NotImplementedError

        assert joint_mask.shape == motion.shape

        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name, joint_mask, face_motion

class Text2MotionDatasetMotionX_VQToken(data.Dataset):
    def __init__(
        self,
        cfg, 
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type,
        input_format, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        # import pdb; pdb.set_trace() 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length

        self.max_motion_token_length = max_motion_length // unit_length + 2 # 51
        self.codebook_size = cfg.model.motion_vae.params.nb_code # 512
        self.mot_end_idx = self.codebook_size # 
        self.mot_pad_idx = self.codebook_size + 1
        # import pdb; pdb.set_trace()
        njoints = get_njoints(dataset_name)

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24





        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        # import pdb; pdb.set_trace()
        count = 0
        bad_count = 0
        miss_count = 0
        
        odd_token_count = 0

        miss_VQ_count = 0

        new_name_list = []
        length_list = []

        token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}')
        # import pdb; pdb.set_trace()
        for i, name in enumerator:
            if count > maxdata:
                break
            
            # try:
            motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))

            # import pdb; pdb.set_trace()
            if input_format == 'root_position':
                motion = motion[..., :4+(njoints-1)*3]
            elif input_format == 'root_position_vel':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # import pdb; pdb.set_trace()
            elif input_format == 'root_position_rot6d':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
            elif input_format == 'root_rot6d':
                motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
            elif input_format in ['all', 'smplx_212']:
                pass
            elif input_format == 'root_body_pos_vel_hand_all':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos_vel':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos':
                motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_rot':
                motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
            elif input_format == 'root_position_vel_only_body':
                motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
            elif input_format == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
                body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
                left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1) # 45
                right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1) # 45

                body_vel_motion = motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
                left_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1)
                right_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1)

                motion = np.concatenate((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)
            else:
                print('NotImplementedError')
                raise NotImplementedError
            
            if 'humanml' in name:
                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                200):
                    bad_count += 1
                    continue
            else:
                if (len(motion)) < min_motion_len:
                    bad_count += 1
                    continue
                elif len(motion) >= self.max_motion_length:
                    start = random.randint(0,len(motion) - self.max_motion_length)
                    motion = motion[start:start+self.max_motion_length]

            motion_token_path = pjoin(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/humanvq', name + ".npy")
            if motion_token_path not in token_all_path:
                miss_VQ_count += 1
                continue

            motion_token = np.load(motion_token_path)
            
            text_data = []
            flag = False
            with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    if 'humanml' in name:
                        if text_source == 'token':
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        bad_count += 1
                                        continue

                                    if int(f_tag*20/unit_length) < int(to_tag*20/unit_length):
                                        motion_token_new = [m_tokens[int(f_tag*20/unit_length) : int(to_tag*20/unit_length)] for m_tokens in motion_token]
                                    else:
                                        motion_token_new = []

                                    if len(motion_token_new) == 0:
                                        continue

                                    if int(to_tag*20/2) - int(f_tag*20/2) != 2 * (int(to_tag*20/4) - int(f_tag*20/4)):
                                        odd_token_count += 1
                                        continue

                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict], 
                                                        "motion_vq_token": motion_token_new}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        elif text_source in ['only_text_token',  'caption']:
                            
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        bad_count += 1
                                        continue

                                    if int(f_tag*20/unit_length) < int(to_tag*20/unit_length):
                                        motion_token_new = [m_tokens[int(f_tag*20/unit_length) : int(to_tag*20/unit_length)] for m_tokens in motion_token]
                                    else:
                                        motion_token_new = []

                                    if len(motion_token_new) == 0:
                                        continue

                                    if int(to_tag*20/2) - int(f_tag*20/2) != 2 * (int(to_tag*20/4) - int(f_tag*20/4)):
                                        odd_token_count += 1
                                        continue

                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict], 
                                                        "motion_vq_token": motion_token_new}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        else:
                            raise NotImplementedError


                    else:
                        text_dict = {}
                        line_split = line.strip()
                        caption = line_split
                        tokens = caption.split(' ')
                        f_tag = 0.0 
                        to_tag = 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)



            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data, 
                                    "motion_vq_token": motion_token}
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            # except:
            #     # import pdb; pdb.set_trace()
            #     miss_count += 1
            #     pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        print(f'Here are {miss_VQ_count} not in VQ')
        print(f'Here are {odd_token_count} odd token_count!')
        # import pdb; pdb.set_trace()
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        # import pdb; pdb.set_trace()

        print('train len', len(data_dict))
        print('test len', len(data_dict))
        # import pdb; pdb.set_trace()
        
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]

        motion, m_length, text_list, motion_token_list = data["motion"], data["length"], data[
            "text"], data['motion_vq_token']
        # Randomly select a caption
        text_data = random.choice(text_list)
        motion_token = random.choice(motion_token_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
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

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)


        m_tokens_len = motion_token.shape[0]

        if m_tokens_len+1 < self.max_motion_token_length:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_token_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name, m_tokens, m_tokens_len

'''For Motion-X'''
class Text2MotionDatasetMotionX_dual_codebook(data.Dataset):
    def __init__(
        self,
        cfg, 
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type,
        input_format, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        # import pdb; pdb.set_trace() 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length

        assert cfg.model.vae_type in ['hvq_body_hand']

        self.unit_length_b = 2
        self.unit_length_t = 4

        self.max_motion_token_length = 98 + 49 + 2

        self.codebook_size = cfg.model.motion_vae.params.nb_code

        self.mot_end_idx = self.codebook_size
        self.mot_pad_idx = self.codebook_size + 1

        # import pdb; pdb.set_trace()
        njoints = get_njoints(dataset_name)

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24





        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        # import pdb; pdb.set_trace()
        count = 0
        bad_count = 0
        length_bad_count = 0
        miss_count = 0
        miss_VQ_count = 0
        odd_token_count = 0

        token_all_path = findAllFile(f'./datasets/Motion-X/vq_tokens/{cfg.model.vae_type}')
        
        new_name_list = []
        length_list = []

        # import pdb; pdb.set_trace()


        for i, name in enumerator:
            if count > maxdata:
                break
            
            try:
                motion_path = pjoin('/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_body_hand_VAE_motionx_feats_ref_norm_back', name + '.npy')
                if os.path.exists(motion_path):
                    motion = np.load(motion_path)
                else:
                    # print('------------')
                    bad_count += 1
                    continue

                # import pdb; pdb.set_trace()
                # if input_format == 'root_position':
                #     motion = motion[..., :4+(njoints-1)*3]
                # elif input_format == 'root_position_vel':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                #     # import pdb; pdb.set_trace()
                # elif input_format == 'root_position_rot6d':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                # elif input_format == 'root_rot6d':
                #     motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                # elif input_format in ['all', 'smplx_212']:
                #     pass
                # elif input_format == 'root_body_pos_vel_hand_all':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos_vel':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_rot':
                #     motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                # elif input_format == 'root_position_vel_only_body':
                #     motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
                #     body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
                #     left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1) # 45
                #     right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1) # 45

                #     body_vel_motion = motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
                #     left_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1)
                #     right_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1)

                #     motion = np.concatenate((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)
                # else:
                #     print('NotImplementedError')
                #     raise NotImplementedError
                 
                # if 'humanml' in name:
                if (len(motion)) < min_motion_len or (len(motion) >=200):
                    # print('++++++++++')
                    length_bad_count += 1
                    continue
                # else:
                #     if (len(motion)) < min_motion_len:
                #         continue
                #     elif len(motion) >= self.max_motion_length:
                #         start = random.randint(0,len(motion) - self.max_motion_length)
                #         motion = motion[start:start+self.max_motion_length]


                motion_token_path_codebook1 = pjoin(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}/motion_vqcode_b', name + ".npy")
                motion_token_path_codebook2 = pjoin(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}/motion_vqcode_t', name + ".npy")

                if motion_token_path_codebook1 not in token_all_path or motion_token_path_codebook2 not in token_all_path:
                    miss_VQ_count += 1
                    continue

                motion_token_codebook1 = np.load(motion_token_path_codebook1)
                motion_token_codebook2 = np.load(motion_token_path_codebook2)

                text_data = []
                flag = False
                with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        if 'humanml' in name:
                            if text_source == 'token':
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            length_bad_count += 1
                                            # print('***************')
                                            continue

                                        motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1 if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b)]
                                        motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2 if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t)]

                                        if len(motion_token_new_codebook1) == 0 or len(motion_token_new_codebook2) == 0:
                                            continue

                                        if int(to_tag*20/self.unit_length_b) - int(f_tag*20/self.unit_length_b) != 2 * (int(to_tag*20/self.unit_length_t) - int(f_tag*20/self.unit_length_t)):
                                            odd_token_count += 1
                                            continue

                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict],
                                                            "motion_vq_token_codebook1": motion_token_new_codebook1, 
                                                            "motion_vq_token_codebook2": motion_token_new_codebook2}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        import pdb; pdb.set_trace()
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            elif text_source in ['only_text_token',  'caption']:
                                
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                                
                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            # pritn('*****************')
                                            length_bad_count += 1
                                            continue

                                        motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1 if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b)]
                                        motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2 if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t)]

                                        if len(motion_token_new_codebook1) == 0 or len(motion_token_new_codebook2) == 0:
                                            continue

                                        # print(motion_token_new_codebook1[0].shape)
                                        # print(motion_token_new_codebook2[0].shape)

                                        if motion_token_new_codebook1[0].shape[0] != 2 * motion_token_new_codebook2[0].shape[0]:
                                            odd_token_count += 1
                                            continue

                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict], 
                                                            "motion_vq_token_codebook1": motion_token_new_codebook1, 
                                                            "motion_vq_token_codebook2": motion_token_new_codebook2}

                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        import pdb; pdb.set_trace()
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            else:
                                raise NotImplementedError
   

                        else:
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split
                            tokens = caption.split(' ')
                            f_tag = 0.0 
                            to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)



                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data, 
                                       "motion_vq_token_codebook1": motion_token_codebook1,
                                       "motion_vq_token_codebook2": motion_token_codebook2}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {bad_count} not in dataset!')
        print(f'Here are {length_bad_count} either small or large than given value.')

        print(f'Here are {miss_VQ_count} not in VQ')

        print(f'Hare are {odd_token_count} odd codebook1_tokens!')

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]

        print('train len', len(data_dict))
        print('test len', len(data_dict))
        
        
        for name in name_list:
            data_codebook1 = random.choice(data_dict[name]['motion_vq_token_codebook1'])
            data_codebook2 = random.choice(data_dict[name]['motion_vq_token_codebook2'])

            try:
                if data_codebook1.shape[0] != 2 * data_codebook2.shape[0]:
                    print('***************************')
                    print(name)
                    import pdb; pdb.set_trace()
            except:
                print('error')
                import pdb; pdb.set_trace()
    
        print('load done')

        # import pdb; pdb.set_trace()
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def compose_new_list(self, list_1, list_2):
        # print('******************')
        # print('list1: {}, list2: {}'.format(len(list_1), len(list_2)))
        
        assert len(list_1) == 2 * len(list_2)

        new_list = []

        for i in range(len(list_1)):
            new_list.append(list_1[i])
            if i % 2 == 1:
                new_list.append(list_2[i // 2])

        return new_list

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]

        motion, m_length, text_list, motion_token_list_codebook1, motion_token_list_codebook2 = data['motion'], data['length'], data['text'], data['motion_vq_token_codebook1'], data['motion_vq_token_codebook2']
        
        m_token_list = []
        for i in range(len(motion_token_list_codebook1)):
            m_token_list_i = self.compose_new_list(motion_token_list_codebook1[i].tolist(), motion_token_list_codebook2[i].tolist())
            m_token_list.append(m_token_list_i)


        m_token_list = np.array(m_token_list)
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        motion_token = random.choice(m_token_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        # if self.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)


        m_tokens_len = motion_token.shape[0]

        if m_tokens_len+1 < self.max_motion_token_length:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_token_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name, m_tokens, m_tokens_len




class Text2MotionDatasetMotionX_body_hand_codebook(data.Dataset):
    def __init__(
        self,
        cfg, 
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type,
        input_format, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        # import pdb; pdb.set_trace() 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length

        import pdb; pdb.set_trace()
        assert cfg.model.vae_type in ['dual_human_vq']

        self.unit_length_b = 4
        self.unit_length_t = 4

        self.max_motion_token_length = 49 + 49 + 2

        self.codebook_size = cfg.model.motion_vae.params.nb_code

        self.mot_end_idx = self.codebook_size
        self.mot_pad_idx = self.codebook_size + 1

        # import pdb; pdb.set_trace()
        njoints = get_njoints(dataset_name)

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24





        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        # import pdb; pdb.set_trace()
        count = 0
        bad_count = 0
        length_bad_count = 0
        miss_count = 0
        miss_VQ_count = 0
        odd_token_count = 0

        body_token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}')
        hand_token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}')
        # token_all_path = findAllFile(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}')

        new_name_list = []
        length_list = []

        # import pdb; pdb.set_trace()


        for i, name in enumerator:
            if count > maxdata:
                break
            
            try:
                motion_path = pjoin('/comp_robot/lushunlin/visualization/visualization/test_case/h2vq_body_hand_VAE_motionx_feats_ref_norm_back', name + '.npy')
                if os.path.exists(motion_path):
                    motion = np.load(motion_path)
                else:
                    # print('------------')
                    bad_count += 1
                    continue

                # import pdb; pdb.set_trace()
                # if input_format == 'root_position':
                #     motion = motion[..., :4+(njoints-1)*3]
                # elif input_format == 'root_position_vel':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                #     # import pdb; pdb.set_trace()
                # elif input_format == 'root_position_rot6d':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                # elif input_format == 'root_rot6d':
                #     motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                # elif input_format in ['all', 'smplx_212']:
                #     pass
                # elif input_format == 'root_body_pos_vel_hand_all':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos_vel':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos':
                #     motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_rot':
                #     motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                # elif input_format == 'root_position_vel_only_body':
                #     motion = np.concatenate((motion[..., :4+(22 - 1) * 3], motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=-1)
                # elif input_format == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
                #     body_pos_motion = motion[..., :4+(22 - 1) * 3] # 67
                #     left_hand_pos_motion = (motion[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1) # 45
                #     right_hand_pos_motion = (motion[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(-1, 15, 3) - body_pos_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1) # 45

                #     body_vel_motion = motion[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
                #     left_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -6:-3][:, None, :]).reshape(motion.shape[0], -1)
                #     right_hand_vel_motion = (motion[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(-1, 15, 3) - body_vel_motion[..., -3:][:, None, :]).reshape(motion.shape[0], -1)

                #     motion = np.concatenate((body_pos_motion, left_hand_pos_motion, right_hand_pos_motion, body_vel_motion, left_hand_vel_motion, right_hand_vel_motion), axis=-1)
                # else:
                #     print('NotImplementedError')
                #     raise NotImplementedError
                 
                # if 'humanml' in name:
                if (len(motion)) < min_motion_len or (len(motion) >=200):
                    # print('++++++++++')
                    length_bad_count += 1
                    continue
                # else:
                #     if (len(motion)) < min_motion_len:
                #         continue
                #     elif len(motion) >= self.max_motion_length:
                #         start = random.randint(0,len(motion) - self.max_motion_length)
                #         motion = motion[start:start+self.max_motion_length]


                motion_token_path_codebook1 = pjoin(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}/motion_vqcode_b', name + ".npy")
                motion_token_path_codebook2 = pjoin(f'/comp_robot/lushunlin/datasets/Motion-X/vq_tokens/{cfg.model.vae_type}/motion_vqcode_t', name + ".npy")

                if motion_token_path_codebook1 not in token_all_path or motion_token_path_codebook2 not in token_all_path:
                    miss_VQ_count += 1
                    continue

                motion_token_codebook1 = np.load(motion_token_path_codebook1)
                motion_token_codebook2 = np.load(motion_token_path_codebook2)

                text_data = []
                flag = False
                with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        if 'humanml' in name:
                            if text_source == 'token':
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            length_bad_count += 1
                                            # print('***************')
                                            continue

                                        motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1 if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b)]
                                        motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2 if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t)]

                                        if len(motion_token_new_codebook1) == 0 or len(motion_token_new_codebook2) == 0:
                                            continue

                                        if int(to_tag*20/self.unit_length_b) - int(f_tag*20/self.unit_length_b) != 2 * (int(to_tag*20/self.unit_length_t) - int(f_tag*20/self.unit_length_t)):
                                            odd_token_count += 1
                                            continue

                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict],
                                                            "motion_vq_token_codebook1": motion_token_new_codebook1, 
                                                            "motion_vq_token_codebook2": motion_token_new_codebook2}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        import pdb; pdb.set_trace()
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            elif text_source in ['only_text_token',  'caption']:
                                
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                                
                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            # pritn('*****************')
                                            length_bad_count += 1
                                            continue

                                        motion_token_new_codebook1 = [m_tokens[int(f_tag*20/self.unit_length_b) : int(to_tag*20/self.unit_length_b)] for m_tokens in motion_token_codebook1 if int(f_tag*20/self.unit_length_b) < int(to_tag*20/self.unit_length_b)]
                                        motion_token_new_codebook2 = [m_tokens[int(f_tag*20/self.unit_length_t) : int(to_tag*20/self.unit_length_t)] for m_tokens in motion_token_codebook2 if int(f_tag*20/self.unit_length_t) < int(to_tag*20/self.unit_length_t)]

                                        if len(motion_token_new_codebook1) == 0 or len(motion_token_new_codebook2) == 0:
                                            continue

                                        # print(motion_token_new_codebook1[0].shape)
                                        # print(motion_token_new_codebook2[0].shape)

                                        if motion_token_new_codebook1[0].shape[0] != 2 * motion_token_new_codebook2[0].shape[0]:
                                            odd_token_count += 1
                                            continue

                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict], 
                                                            "motion_vq_token_codebook1": motion_token_new_codebook1, 
                                                            "motion_vq_token_codebook2": motion_token_new_codebook2}

                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        import pdb; pdb.set_trace()
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            else:
                                raise NotImplementedError
   

                        else:
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split
                            tokens = caption.split(' ')
                            f_tag = 0.0 
                            to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)



                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data, 
                                       "motion_vq_token_codebook1": motion_token_codebook1,
                                       "motion_vq_token_codebook2": motion_token_codebook2}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {bad_count} not in dataset!')
        print(f'Here are {length_bad_count} either small or large than given value.')

        print(f'Here are {miss_VQ_count} not in VQ')

        print(f'Hare are {odd_token_count} odd codebook1_tokens!')

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]

        print('train len', len(data_dict))
        print('test len', len(data_dict))
        
        
        for name in name_list:
            data_codebook1 = random.choice(data_dict[name]['motion_vq_token_codebook1'])
            data_codebook2 = random.choice(data_dict[name]['motion_vq_token_codebook2'])

            try:
                if data_codebook1.shape[0] != 2 * data_codebook2.shape[0]:
                    print('***************************')
                    print(name)
                    import pdb; pdb.set_trace()
            except:
                print('error')
                import pdb; pdb.set_trace()
    
        print('load done')

        # import pdb; pdb.set_trace()
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def compose_new_list(self, list_1, list_2):
        # print('******************')
        # print('list1: {}, list2: {}'.format(len(list_1), len(list_2)))
        
        assert len(list_1) == 2 * len(list_2)

        new_list = []

        for i in range(len(list_1)):
            new_list.append(list_1[i])
            if i % 2 == 1:
                new_list.append(list_2[i // 2])

        return new_list

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx]

        motion, m_length, text_list, motion_token_list_codebook1, motion_token_list_codebook2 = data['motion'], data['length'], data['text'], data['motion_vq_token_codebook1'], data['motion_vq_token_codebook2']
        
        m_token_list = []
        for i in range(len(motion_token_list_codebook1)):
            m_token_list_i = self.compose_new_list(motion_token_list_codebook1[i].tolist(), motion_token_list_codebook2[i].tolist())
            m_token_list.append(m_token_list_i)


        m_token_list = np.array(m_token_list)
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        motion_token = random.choice(m_token_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        # if self.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)


        m_tokens_len = motion_token.shape[0]

        if m_tokens_len+1 < self.max_motion_token_length:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_token_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name, m_tokens, m_tokens_len

'''For Motion-X'''
class Text2MotionDatasetMotionX_text_all(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type, 
        condition, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length
        self.condition = condition
        assert self.condition in ['text_all', 'text_body', 'text_hand', 'text_face', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24


        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]
        # if opt.debug:
        #     id_list = id_list[:100]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        # import pdb; pdb.set_trace()
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        
        count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            
            try:
                motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))

                if (len(motion)) < min_motion_len:
                    continue
                elif len(motion) >= self.max_motion_length:
                    start = random.randint(0,len(motion) - self.max_motion_length)
                    motion = motion[start:start+self.max_motion_length]
                text_data = []
                flag = False
                with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                    
                    try:
                        face_f = open(pjoin(face_text_dir, name + '.txt'))
                        face_text = face_f.readlines()[0]
                    except:
                        import pdb; pdb.set_trace()

                    # import pdb; pdb.set_trace()
                    with open(pjoin(face_text_dir.replace('face_texts', 'body_texts'), name + '.json'), 'r') as body_f:
                        body_dict = json.load(body_f)

                    with open(pjoin(face_text_dir.replace('face_texts', 'hand_texts'), name + '.json'), 'r') as hand_f:
                        hand_dict = json.load(hand_f)


                    for line in f.readlines():
                        if 'humanml' in name:
                            if text_source == 'token':
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                text_dict['face'] = face_text
                                text_dict['hand'] = hand_dict
                                text_dict['body'] = body_dict

                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            elif text_source in ['only_text_token',  'caption']:
                                
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                text_dict['face'] = face_text
                                text_dict['hand'] = hand_dict
                                text_dict['body'] = body_dict

                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            else:
                                raise NotImplementedError
   

                        else:
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split # 'sad, happy'
                            tokens = caption.split(' ') # ['sad', 'happy']
                            f_tag = 0.0 
                            to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            text_dict['face'] = face_text
                            text_dict['hand'] = hand_dict
                            text_dict['body'] = body_dict

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)



                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                import pdb; pdb.set_trace()
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        # import pdb; pdb.set_trace()
        
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        face_text, hand_dict, body_dict = text_data['face'], text_data["hand"],  text_data["body"]

        select_frame = random.randint(0, len(hand_dict)-1)
        hand_frame_data, body_frame_data = hand_dict[str(select_frame)], body_dict[str(select_frame)]

        body_text = random.choice(body_frame_data.split('.')[:-1]) + '.'
        hand_text = random.choice(hand_frame_data.split('.')[:-1]) + '.'

        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
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

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)

        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), body_text, hand_text, face_text