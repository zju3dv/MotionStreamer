# Reference: https://github.com/Li-xingXiao/272-dim-Motion-Representation

# representation: 272 dim
# :2 local xz velocities of root, no heading, can recover translation
# 2:8  heading angular velocities, 6d rotation, can recover heading
# 8:8+3*njoint local position, no heading, all at xz origin
# 8+3*njoint:8+6*njoint local velocities, no heading, all at xz origin, can recover local postion
# 8+6*njoint:8+12*njoint local rotations, 6d rotation, no heading, all frames z+

import numpy as np
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import torch
import os
import visualization.plot_3d_global as plot_3d
import argparse
import tqdm
# from visualization.smplx2joints import process_smplx_data

def findAllFile(base, endswith='.npy'):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.endswith(endswith):
                file_path.append(fullname)
    return file_path

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1

    return -heading, axis

def accumulate_rotations(relative_rotations):
    """Accumulate relative rotations to get the overall rotation"""
    # Initial rotation is the rotation matrix
    R_total = [relative_rotations[0]]
    # Iterate through all relative rotations, accumulating them
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):
    # take positions_no_heading: local position on xz ori, no heading
    # velocities_root_xy_no_heading: to recover translation
    # global_heading_diff_rot: to recover root rotation
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:,8:8+3*njoint].reshape(nfrm, -1, 3) # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
    global_heading_diff_rot = final_x[:,2:8] # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), positions_no_heading[...,None]).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


# add hip height to translation when recoverring from rotation
def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])
    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85

def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    smpl_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smpl_85




def smpl85_2_smpl322(smpl_85_data):
    result = np.concatenate((smpl_85_data[:,:66], np.zeros((smpl_85_data.shape[0], 90)), np.zeros((smpl_85_data.shape[0], 3)), np.zeros((smpl_85_data.shape[0], 50)), np.zeros((smpl_85_data.shape[0], 100)), smpl_85_data[:,72:72+3], smpl_85_data[:,75:]), axis=-1)
    return result

def visualize_smpl_85(data, title=None, output_path='visualize_result', name='', fps=30):
    # data: torch.Size([nframe, 85])
    smpl_85_data = data
    if len(smpl_85_data.shape) == 3:
       smpl_85_data = np.squeeze(smpl_85_data, axis=0)
    
    smpl_85_data = smpl85_2_smpl322(smpl_85_data)
    vert, joints, motion, faces = process_smplx_data(smpl_85_data, norm_global_orient=False, transform=False)
    xyz = joints[:, :22, :].reshape(1, -1, 22, 3).detach().cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pose_vis = plot_3d.draw_to_batch(xyz, title_batch=title, outname=[f'{output_path}/rot_{name}.mp4'], fps=fps)
    return output_path


def visualize_pos_xyz(xyz, title_batch=None, output_path='./', name='', fps=30):
    # xyz: torch.Size([nframe, 22, 3])
    xyz = xyz[:1]   
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_xyz = plot_3d.draw_to_batch(xyz, title_batch, [f'{output_path}/pos_{name}.mp4'], fps=fps)
    return output_path


if __name__ == '__main__':
    njoint = 22
    parser = argparse.ArgumentParser(description='Visualize new representation.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input path')
    parser.add_argument('--mode', type=str, required=True, default='rot', choices=['rot', 'pos'], help='Recover from rotation or position')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for data_path in tqdm.tqdm(findAllFile(args.input_dir, endswith='.npy')):
        data_272 = np.load(data_path)
        if args.mode == 'rot':
            # recover from rotation
            global_rotation = recover_from_local_rotation(data_272, njoint)  # get the 85-dim smpl data
            visualize_smpl_85(global_rotation, output_path=args.output_dir, name=data_path.split('/')[-1].split('.')[0])
            print(f"Visualized results are saved in {args.output_dir}")
        else:
            # recover from position
            global_position = recover_from_local_position(data_272, njoint)
            global_position = np.expand_dims(global_position, axis=0)
            visualize_pos_xyz(global_position, output_path=args.output_dir, name=data_path.split('/')[-1].split('.')[0])
            print(f"Visualized results are saved in {args.output_dir}")