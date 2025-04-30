from os.path import join as pjoin
import numpy as np
# from .humanml.utils.word_vectorizer import WordVectorizer, WordVectorizer_only_text_token
from .utils import *
from .HumanML3D_272 import HumanML3D_272_DataModule


def get_mean_std(phase, cfg, dataset_name):
    assert dataset_name == 'humanml3d_272'
        
    data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
    mean = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "Mean.npy"))
    std = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "Std.npy"))
    return mean, std



def get_njoints(dataset_name):
    njoints = 22
    return njoints


def reget_mean_std(cfg, dataset_name, mean, std):
    if 'MINOR_MOTION_TYPE' in cfg.DATASET:
        select_motion_type = cfg.DATASET.MINOR_MOTION_TYPE
    else:
        select_motion_type = cfg.DATASET.MOTION_TYPE
    
    njoints = get_njoints(dataset_name)
    if select_motion_type == 'root_position':
        mean = mean[..., :4+(njoints - 1) * 3]
    elif select_motion_type == 'root_position_vel':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    elif select_motion_type == 'root_position_rot6d':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=0)
    elif select_motion_type == 'root_rot6d':
        mean = np.concatenate((mean[..., :4], mean[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=0)
    elif select_motion_type in ['all', 'smplx_212', 'vector_263', 'vector_263_ori_humanml', 'smplx_159', '']:
        pass
    elif select_motion_type == 'root_body_pos_vel_hand_all':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], mean[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
        # pass
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_rot':
        mean = np.concatenate((mean[..., :4+(22 - 1) * 3], mean[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_position_vel_only_body':
        mean = np.concatenate((mean[..., :4+(22 - 1) * 3], mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
        body_pos_mean = mean[..., :4+(22 - 1) * 3] # 67
        left_hand_pos_mean = (mean[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(15, 3) - body_pos_mean[..., -6:-3]).reshape(-1) # 45
        right_hand_pos_mean = (mean[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(15, 3) - body_pos_mean[..., -3:]).reshape(-1) # 45

        body_vel_mean = mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
        left_hand_vel_mean = (mean[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(15, 3) - body_vel_mean[..., -6:-3]).reshape(-1)
        right_hand_vel_mean = (mean[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(15, 3) - body_vel_mean[..., -3:]).reshape(-1)
        
        mean = np.concatenate((body_pos_mean, left_hand_pos_mean, right_hand_pos_mean, body_vel_mean, left_hand_vel_mean, right_hand_vel_mean), axis=0)
    else:
        raise NotImplementedError
    
    if select_motion_type == 'root_position':
        std = std[..., :4+(njoints-1)*3]
    elif select_motion_type == 'root_position_vel':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    elif select_motion_type == 'root_position_rot6d':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=0)
    elif select_motion_type == 'root_rot6d':
        std = np.concatenate((std[..., :4], std[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=0)
    elif select_motion_type in ['all', 'smplx_212', 'vector_263', 'vector_263_ori_humanml', 'smplx_159', '']:
        pass
    elif select_motion_type == 'root_body_pos_vel_hand_all':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], std[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
        # pass
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_rot':
        std = np.concatenate((std[..., :4+(22 - 1) * 3], std[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], std[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_position_vel_only_body':
        std = np.concatenate((std[..., :4+(22 - 1) * 3], std[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
        std = np.concatenate((std[..., :4+(njoints - 1) * 3], std[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    else:
        raise NotImplementedError

    return mean, std

# def get_WordVectorizer(cfg, phase, dataset_name):
#     if phase not in ["text_only"]:
#         if dataset_name.lower() in ['humanml3d_272']:
#             if cfg.model.eval_text_source == 'token':
#                 return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab", cfg.model.eval_text_encode_way)
#             else:
#                 return WordVectorizer_only_text_token(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab", cfg.model.eval_text_encode_way)
#         else:
#             raise ValueError("Only support WordVectorizer for HumanML3D_272")
#     else:
#         return None


def get_collate_fn(name, cfg, phase="train"):
    if name.lower() in ['humanml3d_272']:
        if cfg.model.condition in ['text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_text_all
        elif cfg.TEST.inference_vq_code:
            return vq_collate
        elif cfg.TRAIN.STAGE in ['gpt'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_vq_token
        else:
            return mld_collate
    else:
        raise NotImplementedError


# map config name to module&path
dataset_module_map = {
    'humanml3d_272': HumanML3D_272_DataModule
}
motion_subdir = {'humanml3d_272': 'motion_data'}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d_272"]:
    
            if 'MINOR_MOTION_TYPE' in cfg.DATASET:
                input_format = cfg.DATASET.MINOR_MOTION_TYPE
            else:
                input_format = cfg.DATASET.MOTION_TYPE
            
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)

            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)
            
            # get WordVectorizer
            # wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)
            # get dataset module
            
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                # w_vectorizer=wordVectorizer,
                input_format=cfg.DATASET.MOTION_TYPE, 
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
                
        else:
            raise NotImplementedError

    if input_format == 'root_body_pos_vel_hand_pos_vel':
        cfg.DATASET.NFEATS = 313
    else:
        cfg.DATASET.NFEATS = datasets[0].nfeats

    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
