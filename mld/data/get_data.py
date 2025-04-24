from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer, WordVectorizer_only_text_token
from .HumanML3D import HumanML3DDataModule
from .Kit import KitDataModule
from .Humanact12 import Humanact12DataModule
from .Uestc import UestcDataModule
from .utils import *
from .MotionX import Motion_XDataModule
from .MotionX_V25 import Motion_X_V25_DataModule
from .MotionX_V26 import Motion_X_V26_DataModule
from .HumanML3D_v3 import HumanML3D_V3_DataModule


def get_mean_std(phase, cfg, dataset_name):
    # if phase == 'gt':
    #     # used by T2M models (including evaluators)
    #     mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    #     std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    # elif phase in ['train', 'val', 'text_only']:
    #     # used by our models
    #     mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    #     std = np.load(pjoin(opt.data_root, 'Std.npy'))
    # import pdb; pdb.set_trace()
    # todo: use different mean and val for phases


    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit", "motionx", "motionx_v25", 'motionx_v26', 'humanml3d_v3']
    # import pdb; pdb.set_trace()
    # if phase in ["train", "val", "test"]:
    if name in ['t2m', 'kit']:
        if phase in ["val"]:
            if name == 't2m':
                data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01",
                                "meta")
            elif name == 'kit':
                data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005",
                                "meta")
            else:
                raise ValueError("Only support t2m and kit")
            mean = np.load(pjoin(data_root, "mean.npy"))
            std = np.load(pjoin(data_root, "std.npy"))
        else:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            mean = np.load(pjoin(data_root, "Mean.npy"))
            std = np.load(pjoin(data_root, "Std.npy"))
    elif name in ['motionx', 'motionx_v25', 'motionx_v26']:

        if phase in ["val"]:
            # import pdb; pdb.set_trace()
            data_root = pjoin(cfg.model.t2m_path, name, cfg.DATASET.VERSION,cfg.DATASET.MOTION_TYPE, "Decomp_SP001_SM001_H512_b8192_lr2e-4",
                            "meta")
            mean = np.load(pjoin(data_root, "mean.npy"))
            std = np.load(pjoin(data_root, "std.npy"))

        else:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            mean = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "mean.npy"))
            std = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "std.npy"))
            
    elif name in ['humanml3d_v3']:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "Mean.npy"))
        std = np.load(pjoin(data_root, 'mean_std', cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, "Std.npy"))
        
    else:
        raise NotImplementedError

    return mean, std



def get_njoints(dataset_name):
    if dataset_name in ['humanml3d', 'humanml3d_v3']:
        njoints = 22
    elif dataset_name == 'kit':
        njoints = 21
    elif dataset_name in ['motionx', 'motionx_v25', 'motionx_v26']:
        njoints = 52
    else:
        raise NotImplementedError
    
    return njoints



def reget_mean_std(cfg, dataset_name, mean, std):
    # import pdb; pdb.set_trace()
    if 'MINOR_MOTION_TYPE' in cfg.DATASET:
        select_motion_type = cfg.DATASET.MINOR_MOTION_TYPE
    else:
        select_motion_type = cfg.DATASET.MOTION_TYPE
    
    njoints = get_njoints(dataset_name)
    # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 3 + 21 * 6 : 4+(njoints - 1) * 9], mean[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
        # pass
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos':
        mean = np.concatenate((mean[..., :4+(njoints - 1) * 3], mean[..., 4+(njoints - 1) * 9 + 22 * 3: 4+(njoints - 1) * 9 + 52*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_rot':
        # body_pos 4:4+(22 - 1) * 3
        # hand_rot: 4+(52 - 1) * 3 + （22-1）*6 ： 4+（52-1）*9
        # body vel: 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3
        # import pdb; pdb.set_trace()
        mean = np.concatenate((mean[..., :4+(22 - 1) * 3], mean[..., 4+(52 - 1) * 3 + (22-1)*6 : 4+(52-1)*9], mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_position_vel_only_body':
        mean = np.concatenate((mean[..., :4+(22 - 1) * 3], mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3]), axis=0)
    elif select_motion_type == 'root_body_pos_vel_hand_pos_vel_hand_wrist':
        # import pdb; pdb.set_trace()
        body_pos_mean = mean[..., :4+(22 - 1) * 3] # 67
        left_hand_pos_mean = (mean[..., 4+(22 - 1) * 3:4+(37 - 1) * 3].reshape(15, 3) - body_pos_mean[..., -6:-3]).reshape(-1) # 45
        right_hand_pos_mean = (mean[..., 4+(37 - 1) * 3:4+(52 - 1) * 3].reshape(15, 3) - body_pos_mean[..., -3:]).reshape(-1) # 45

        body_vel_mean = mean[..., 4+(52 - 1) * 9: 4+(52 - 1) * 9 + 22*3] # 66
        left_hand_vel_mean = (mean[..., 4+(52 - 1) * 9 + 22*3: 4+(52 - 1) * 9 + 22*3 + 15 * 3].reshape(15, 3) - body_vel_mean[..., -6:-3]).reshape(-1)
        right_hand_vel_mean = (mean[..., 4+(52 - 1) * 9 + 22*3+ 15 * 3: 4+(52 - 1) * 9 + 22*3 + 15 * 3 + 15 * 3].reshape(15, 3) - body_vel_mean[..., -3:]).reshape(-1)
        
        mean = np.concatenate((body_pos_mean, left_hand_pos_mean, right_hand_pos_mean, body_vel_mean, left_hand_vel_mean, right_hand_vel_mean), axis=0)
    else:
        raise NotImplementedError
    
    # import pdb; pdb.set_trace()
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

def get_WordVectorizer(cfg, phase, dataset_name):
    # import pdb; pdb.set_trace()
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["motionx", "motionx_v25", "humanml3d", "kit", 'motionx_v26', 'humanml3d_v3']:
            if cfg.model.eval_text_source == 'token':
                return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab", cfg.model.eval_text_encode_way)
            else:
                return WordVectorizer_only_text_token(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab", cfg.model.eval_text_encode_way)
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, cfg, phase="train"):
    # import pdb; pdb.set_trace()
    if name.lower() in ["humanml3d", "kit", 'humanml3d_v3']:
        if cfg.model.condition in ['text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_text_all
        elif cfg.TEST.inference_vq_code:
            return vq_collate
        elif cfg.TRAIN.STAGE in ['gpt'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_vq_token
        else:
            return mld_collate
    elif name.lower() in ["motionx", 'motionx_v25', 'motionx_v26']:
        if cfg.model.condition in ['text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_text_all
        elif cfg.TEST.inference_vq_code:
            return vq_collate
        elif cfg.TRAIN.STAGE in ['gpt'] and (not cfg.TEST.inference_vq_code):
            return mld_collate_vq_token
        elif cfg.model.vae_type == 'hvq_body_hand_face':
            return mld_motionx_with_face_collate
        else:
            return mld_motionx_collate

    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate
    else:
        raise NotImplementedError
    # else:
    #     return all_collate
    # if phase == "test":
    #     return eval_collate
    # else:


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "kit": KitDataModule,
    "humanact12": Humanact12DataModule,
    "uestc": UestcDataModule,
    "motionx": Motion_XDataModule, 
    'motionx_v25': Motion_X_V25_DataModule, 
    'motionx_v26': Motion_X_V26_DataModule, 
    'humanml3d_v3': HumanML3D_V3_DataModule
}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs", "motionx": "motion_data", "motionx_v25": "motion_data", 'motionx_v26': "motion_data", 'humanml3d_v3': 'motion_data'}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg

    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    # import pdb; pdb.set_trace()
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)

            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)
            # import pdb; pdb.set_trace()
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)
            # get dataset module
            # import pdb; pdb.set_trace()
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
                w_vectorizer=wordVectorizer,
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
        elif dataset_name.lower() in ["humanact12", 'uestc']:
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                num_frames=cfg.DATASET.HUMANACT12.NUM_FRAMES,
                sampling=cfg.DATASET.SAMPLER.SAMPLING,
                sampling_step=cfg.DATASET.SAMPLER.SAMPLING_STEP,
                pose_rep=cfg.DATASET.HUMANACT12.POSE_REP,
                max_len=cfg.DATASET.SAMPLER.MAX_LEN,
                min_len=cfg.DATASET.SAMPLER.MIN_LEN,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                glob=cfg.DATASET.HUMANACT12.GLOB,
                translation=cfg.DATASET.HUMANACT12.TRANSLATION)
            cfg.DATASET.NCLASSES = dataset.nclasses
            datasets.append(dataset)
        elif dataset_name.lower() in ["amass"]:
            # todo: add amass dataset
            raise NotImplementedError
        
        elif dataset_name.lower() in ["motionx"]:

            if 'MINOR_MOTION_TYPE' in cfg.DATASET:
                input_format = cfg.DATASET.MINOR_MOTION_TYPE
            else:
                input_format = cfg.DATASET.MOTION_TYPE

            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # import pdb; pdb.set_trace()

            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)

            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)
            # import pdb; pdb.set_trace()
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
                w_vectorizer=wordVectorizer,
                semantic_text_dir=cfg.DATASET.MOTIONX.SEMANTIC_TEXT_ROOT,
                face_text_dir= cfg.DATASET.MOTIONX.FACE_TEXT_ROOT, 
                condition = cfg.model.condition, 
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                dataset_name = dataset_name,
                eval_text_encode_way = cfg.model.eval_text_encode_way, 
                text_source = cfg.DATASET.TEXT_SOURCE, 
                motion_type = cfg.DATASET.MOTION_TYPE, 
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                input_format = input_format, 
            )
            datasets.append(dataset)


        elif dataset_name.lower() in ["motionx_v25"]:
            # import pdb; pdb.set_trace()
            if 'MINOR_MOTION_TYPE' in cfg.DATASET:
                input_format = cfg.DATASET.MINOR_MOTION_TYPE
            else:
                input_format = cfg.DATASET.MOTION_TYPE

            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # import pdb; pdb.set_trace()

            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)

            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
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
                w_vectorizer=wordVectorizer,
                semantic_text_dir=cfg.DATASET.MOTIONX_V25.SEMANTIC_TEXT_ROOT,
                face_text_dir= cfg.DATASET.MOTIONX_V25.FACE_TEXT_ROOT, 
                condition = cfg.model.condition, 
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                dataset_name = dataset_name,
                eval_text_encode_way = cfg.model.eval_text_encode_way, 
                text_source = cfg.DATASET.TEXT_SOURCE, 
                motion_type = cfg.DATASET.MOTION_TYPE, 
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                input_format = input_format, 
            )
            datasets.append(dataset)


        elif dataset_name.lower() in ["motionx_v26"]:
            # import pdb; pdb.set_trace()
            if 'MINOR_MOTION_TYPE' in cfg.DATASET:
                input_format = cfg.DATASET.MINOR_MOTION_TYPE
            else:
                input_format = cfg.DATASET.MOTION_TYPE

            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # import pdb; pdb.set_trace()

            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)

            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
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
                w_vectorizer=wordVectorizer,
                semantic_text_dir=cfg.DATASET.MOTIONX_V26.SEMANTIC_TEXT_ROOT,
                face_text_dir= cfg.DATASET.MOTIONX_V26.FACE_TEXT_ROOT, 
                condition = cfg.model.condition, 
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                dataset_name = dataset_name,
                eval_text_encode_way = cfg.model.eval_text_encode_way, 
                text_source = cfg.DATASET.TEXT_SOURCE, 
                motion_type = cfg.DATASET.MOTION_TYPE, 
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                input_format = input_format, 
                hand_mask = cfg.LOSS.hand_mask
            )
            datasets.append(dataset)

        elif dataset_name.lower() in ["humanml3d_v3"]:
    
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
            # import pdb; pdb.set_trace()
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)
            # get dataset module
            # import pdb; pdb.set_trace()
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
                w_vectorizer=wordVectorizer,
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
    # import pdb; pdb.set_trace()
    if input_format == 'root_body_pos_vel_hand_pos_vel':
        cfg.DATASET.NFEATS = 313
    else:
        cfg.DATASET.NFEATS = datasets[0].nfeats

    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
