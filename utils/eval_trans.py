import numpy as np
import torch
from scipy import linalg

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