import numpy as np
import torch
from scipy import linalg
from utils.face_z_align_util import rotation_6d_to_matrix
import visualization.plot_3d_global as plot_3d
import os

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, title_batch=None, outname=None, fps=30):
    xyz = xyz[:1]   
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = fps)

def calculate_mpjpe(gt_joints, pred_joints):
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) 
    mpjpe_seq = mpjpe.mean(-1)

    return mpjpe_seq


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):

    if final_x.ndim == 3:
        bs, nfrm, _ = final_x.shape
        is_batched = True
    else:
        nfrm, _ = final_x.shape
        bs = 1
        is_batched = False
        final_x = final_x.reshape(1, *final_x.shape)

    
    positions_no_heading = final_x[:,:,8:8+3*njoint].reshape(bs, nfrm, njoint, 3) 
    velocities_root_xy_no_heading = final_x[:,:,:2] 
    global_heading_diff_rot = final_x[:,:,2:8] 

   
    positions_with_heading = []
    for b in range(bs):
        
        global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot[b])).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        
        
        curr_pos_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), 
                                        positions_no_heading[b][...,None]).squeeze(-1)

        
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading[b].shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[b, :, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[b, :, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], 
                                                         velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

        
        curr_pos_with_heading[:, :, 0] += root_translation[:, 0:1]
        curr_pos_with_heading[:, :, 2] += root_translation[:, 2:]
        
        positions_with_heading.append(curr_pos_with_heading)

    positions_with_heading = np.stack(positions_with_heading, axis=0)

    if not is_batched:
        positions_with_heading = positions_with_heading.squeeze(0)

    return positions_with_heading

@torch.no_grad()                
def evaluation_gt(val_loader, evaluator, device=torch.device('cuda')):  
    textencoder, motionencoder = evaluator
    motion_annotation_list = []
    R_precision_real = torch.tensor([0,0,0], device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    nb_sample = torch.tensor(0, device=device)
    
    for batch in val_loader:
        text, pose, m_length = batch
        pose = pose.to(device).float()
        et, em = textencoder(text).loc, motionencoder(pose, m_length).loc
        motion_annotation_list.append(em)
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += torch.tensor(temp_R, device=device)
        matching_score_real += torch.tensor(temp_match, device=device)
        nb_sample += et.shape[0]

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    matching_score_real = matching_score_real / nb_sample
    
    # for GT data, no need to calculate fid
    fid = 0.0
    
    return fid, diversity_real, R_precision_real[0], R_precision_real[1], R_precision_real[2], matching_score_real

# Single-GPU evaluation of Causal TAE (test time)
@torch.no_grad()
def evaluation_tae_single(out_dir, val_loader, net, logger, writer, evaluator, device=torch.device('cuda')): 
    net.eval()
    nb_sample = 0
    
    textencoder, motionencoder = evaluator

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = torch.tensor(0, device=device)
    mpjpe = torch.tensor(0.0, device=device)
    num_poses = torch.tensor(0, device=device)

    for batch in val_loader:
        motion, m_length = batch
        motion = motion.to(device)
        motion = motion.float()
        bs, seq = motion.shape[0], motion.shape[1]
        em = motionencoder(motion, m_length).loc
                    
        num_joints = 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(device)

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_local_position(pose.squeeze(0), num_joints)
            pred_pose, _, _ = net(motion[i:i+1, :m_length[i]])
            
            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose
            
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                
            pred_xyz = recover_from_local_position(pred_denorm.squeeze(0), num_joints)
            pred_xyz = torch.from_numpy(pred_xyz).float().to(device)
            pose_xyz = torch.from_numpy(pose_xyz).float().to(device)
            
            mpjpe += torch.sum(calculate_mpjpe(pose_xyz[:, :m_length[i]].squeeze(), pred_xyz[:, :m_length[i]].squeeze()))
            num_poses += pose_xyz.shape[0]

        em_pred = motionencoder(pred_pose_eval, m_length).loc

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        nb_sample += bs  
    
    mpjpe = mpjpe / num_poses
    mpjpe = mpjpe * 1000   # mm

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. :, FID. {fid:.4f}, mpjpe. {mpjpe:.5f} (mm)"
    logger.info(msg)
    
    return fid, mpjpe, writer, logger

# Multi-GPU evaluation of Causal TAE (training time)
@torch.no_grad()        
def evaluation_tae_multi(out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_mpjpe, draw = True, save = True, savegif = True, device=torch.device('cuda'), accelerator=None): 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []

    nb_sample = torch.tensor(0, device=device)
    mpjpe = torch.tensor(0.0, device=device)
    num_poses = torch.tensor(0, device=device)

    for batch in val_loader:
        motion, m_length = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        num_joints = 22
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(device)

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_local_position(pose.squeeze(0), num_joints)

            pred_pose, _, _ = net(motion[i:i+1, :m_length[i]])
            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if accelerator is None or accelerator.is_main_process:
                pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                pred_xyz = recover_from_local_position(pred_denorm.squeeze(0), num_joints)
                pred_xyz = torch.from_numpy(pred_xyz).float().to(device)
                pose_xyz = torch.from_numpy(pose_xyz).float().to(device)
                mpjpe += torch.sum(calculate_mpjpe(pose_xyz[:, :m_length[i]].squeeze(), pred_xyz[:, :m_length[i]].squeeze()))
                num_poses += pose_xyz.shape[0]

                if i < 4:
                    draw_org.append(pose_xyz)
                    draw_pred.append(pred_xyz)
                    draw_text.append('')
        nb_sample += bs


    if accelerator is not None:
        accelerator.wait_for_everyone()
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        mpjpe = accelerator.reduce(mpjpe, reduction="sum")
        
    if accelerator is None or accelerator.is_main_process:
        mpjpe = mpjpe / num_poses    
        # transform mpjpe to mm
        mpjpe = mpjpe * 1000
        msg = f"--> \t Eva. Iter {nb_iter} :, mpjpe. {mpjpe:.3f} (mm)"
        logger.info(msg)
    
    # save visualization on tensorboard
    if draw and (accelerator is None or accelerator.is_main_process):
        writer.add_scalar('./Test/mpjpe', mpjpe, nb_iter)

        if nb_iter % 20000 == 0 : 
            for ii in range(4):
                draw_org[ii] = draw_org[ii].unsqueeze(0)
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None, fps=30)
            
        if nb_iter % 20000 == 0 : 
            for ii in range(4):
                draw_pred[ii] = draw_pred[ii].unsqueeze(0)
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None, fps=30)   

    if accelerator is None or accelerator.is_main_process:
        if mpjpe < best_mpjpe :
            msg = f"--> --> \t mpjpe Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
            logger.info(msg)
            best_mpjpe = mpjpe
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_iter, best_mpjpe, writer, logger


# Single-GPU evaluation of text to motion model (test time)：
@torch.no_grad()                
def evaluation_transformer_272_single(val_loader, net, trans, tokenize_model, logger, evaluator, cfg=4.0, device=torch.device('cuda'), unit_length=4):     
    textencoder, motionencoder = evaluator
    trans.eval()
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = torch.tensor([0,0,0], device=device)
    R_precision = torch.tensor([0,0,0], device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    matching_score_pred = torch.tensor(0.0, device=device)

    nb_sample = torch.tensor(0, device=device)

    for batch in val_loader:
        text, pose, m_length = batch
        bs, seq = pose.shape[:2]
        num_joints = 22
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(device)
        pred_len = torch.ones(bs).long()
        
        for k in range(bs):    
            index_motion = trans.sample_for_eval_CFG(text[k:k+1], length=m_length[k], tokenize_model=tokenize_model, device=device, unit_length=unit_length, cfg=cfg)
            pred_pose = net.forward_decoder(index_motion)            
            cur_len = pred_pose.shape[1]
            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

        et_pred, em_pred = textencoder(text).loc, motionencoder(pred_pose_eval, pred_len).loc
        
        pose = pose.to(device).float()
        et, em = textencoder(text).loc, motionencoder(pose, m_length).loc
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += torch.tensor(temp_R, device=device)
        matching_score_real += torch.tensor(temp_match, device=device)
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += torch.tensor(temp_R, device=device)
        matching_score_pred += torch.tensor(temp_match, device=device)
        nb_sample += et.shape[0]

        pose = torch.tensor(pose).to(device)
    
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eval. :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity Pred. {diversity:.4f}, R_precision Real. {R_precision_real}, R_precision Pred. {R_precision}, MM-dist (matching_score) Real. {matching_score_real}, MM-dist (matching_score) Pred. {matching_score_pred}"
    logger.info(msg)

    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, logger

def euclidean_distance_matrix(matrix1, matrix2):
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    
    d3 = np.sum(np.square(matrix2), axis=1)     
    dists = np.sqrt(d1 + d2 + d3)
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)  
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace() 
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score



def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)  
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1) 
    return dist.mean()  



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1) 
    mu2 = np.atleast_1d(mu2) 

    sigma1 = np.atleast_2d(sigma1) 
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False) 
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
