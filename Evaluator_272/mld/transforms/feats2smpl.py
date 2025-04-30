from os.path import join as pjoin

import numpy as np
import torch

import mld.data.humanml.utils.paramUtil as paramUtil
from mld.data.humanml.data.dataset import Text2MotionDatasetV2
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.data.humanml.utils.plot_script import plot_3d_motion

skeleton = paramUtil.t2m_kinematic_chain




def main():
    data_root = '../datasets/humanml3d'
    feastures_path = 'in.npy'
    animation_save_path = 'in.mp4'

    fps = 20
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    motion = np.load(feastures_path)
    motion = motion * std + mean
    motion_rec = recover_from_ric(torch.tensor(motion), 22).cpu().numpy()
    # with open('in_22.npy', 'wb') as f:
    #     np.save(f,motion_rec)
    motion_rec = motion_rec * 1.3
    plot_3d_motion(animation_save_path, motion_rec, title='input', fps=fps)


if __name__ == '__main__':
    main()
