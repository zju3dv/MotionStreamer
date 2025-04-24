from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from mld.models.tools.tools import remove_padding
from mld.transforms.joints2jfeats import Rifke
from mld.utils.geometry import matrix_of_angles

from .utils import l2_norm, variance


class ComputeMetrics_body_hand(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = False,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d", "motionx", 'motionx_v26']:
            raise NotImplementedError("This jointstype is not implemented.")
        # import pdb; pdb.set_trace()
        self.name = 'APE and AVE'
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype, normalization=False)

        self.force_in_meter = force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        # import pdb; pdb.set_trace()
        # APE
        self.add_state("APE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")

        self.add_state("APE_pose_body",
                       default=torch.zeros(22 - 1),
                       dist_reduce_fx="sum")

        self.add_state("APE_pose_hand",
                       default=torch.zeros(30),
                       dist_reduce_fx="sum")


        self.add_state("APE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")


        self.add_state("APE_joints_body",
                       default=torch.zeros(22),
                       dist_reduce_fx="sum")

        self.add_state("APE_joints_hand",
                       default=torch.zeros(30),
                       dist_reduce_fx="sum")

        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_pose_body", "APE_pose_hand", "APE_joints", "APE_joints_body", "APE_joints_hand"]

        # AVE
        self.add_state("AVE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")

        self.add_state("AVE_pose_body",
                       default=torch.zeros(22 - 1),
                       dist_reduce_fx="sum")

        self.add_state("AVE_pose_hand",
                       default=torch.zeros(30),
                       dist_reduce_fx="sum")

        self.add_state("AVE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")

        self.add_state("AVE_joints_body",
                       default=torch.zeros(22),
                       dist_reduce_fx="sum")

        self.add_state("AVE_joints_hand",
                       default=torch.zeros(30),
                       dist_reduce_fx="sum")


        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_pose_body", "AVE_pose_hand", "AVE_joints", "AVE_joints_body", "AVE_joints_hand"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self, sanity_flag):
        count = self.count
        APE_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.APE_metrics
        }

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_pose_body"] = self.APE_pose_body.mean() / count
        APE_metrics["APE_mean_pose_hand"] = self.APE_pose_hand.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count
        APE_metrics["APE_mean_joints_body"] = self.APE_joints_body.mean() / count
        APE_metrics["APE_mean_joints_hand"] = self.APE_joints_hand.mean() / count

        # Remove arrays
        APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_pose_body")
        APE_metrics.pop("APE_pose_hand")
        APE_metrics.pop("APE_joints")
        APE_metrics.pop("APE_joints_body")
        APE_metrics.pop("APE_joints_hand")

        count_seq = self.count_seq
        AVE_metrics = {
            metric: getattr(self, metric) / count_seq
            for metric in self.AVE_metrics
        }

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_pose_body"] = self.AVE_pose_body.mean() / count_seq
        AVE_metrics["AVE_mean_pose_hand"] = self.AVE_pose_hand.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq
        AVE_metrics["AVE_mean_joints_body"] = self.AVE_joints_body.mean() / count_seq
        AVE_metrics["AVE_mean_joints_hand"] = self.AVE_joints_hand.mean() / count_seq

        # Remove arrays
        AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_pose_body")
        AVE_metrics.pop("AVE_pose_hand")
        AVE_metrics.pop("AVE_joints")
        AVE_metrics.pop("AVE_joints_body")
        AVE_metrics.pop("AVE_joints_hand")


        return {**APE_metrics, **AVE_metrics}

    def update(self, jts_text: Tensor, jts_ref: Tensor, lengths: List[int]):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        jts_text, poses_text, root_text, traj_text = self.transform(
            jts_text, lengths)
        jts_ref, poses_ref, root_ref, traj_ref = self.transform(
            jts_ref, lengths)

        # import pdb; pdb.set_trace()




        for i in range(len(lengths)):
            # import pdb; pdb.set_trace()
            jts_text_body = jts_text[i][..., :22, :]
            jts_text_hand = jts_text[i][..., 22:, :]
            jts_ref_body = jts_ref[i][..., :22, :]
            jts_ref_hand = jts_ref[i][..., 22:, :]


            poses_text_body = poses_text[i][..., :21, :]
            poses_text_hand = poses_text[i][..., 21:, :]
            poses_ref_body = poses_ref[i][..., :21, :]
            poses_ref_hand = poses_ref[i][..., 21:, :]

            self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            self.APE_pose_body += l2_norm(poses_text_body, poses_ref_body, dim=2).sum(0)
            self.APE_pose_hand += l2_norm(poses_text_hand, poses_ref_hand, dim=2).sum(0)

            self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)
            self.APE_joints_body += l2_norm(jts_text_body, jts_ref_body, dim=2).sum(0)
            self.APE_joints_hand += l2_norm(jts_text_hand, jts_ref_hand, dim=2).sum(0)

            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            poses_body_sigma_text = variance(poses_text_body, lengths[i], dim=0)
            poses_body_sigma_ref = variance(poses_ref_body, lengths[i], dim=0)
            self.AVE_pose_body += l2_norm(poses_body_sigma_text, poses_body_sigma_ref, dim=1)


            poses_hand_sigma_text = variance(poses_text_hand, lengths[i], dim=0)
            poses_hand_sigma_ref = variance(poses_ref_hand, lengths[i], dim=0)
            self.AVE_pose_hand += l2_norm(poses_hand_sigma_text, poses_hand_sigma_ref, dim=1)


            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

            jts_body_sigma_text = variance(jts_text_body, lengths[i], dim=0)
            jts_body_sigma_ref = variance(jts_ref_body, lengths[i], dim=0)
            self.AVE_joints_body += l2_norm(jts_body_sigma_text, jts_body_sigma_ref, dim=1)

            jts_hand_sigma_text = variance(jts_text_hand, lengths[i], dim=0)
            jts_hand_sigma_ref = variance(jts_ref_hand, lengths[i], dim=0)
            self.AVE_joints_hand += l2_norm(jts_hand_sigma_text, jts_hand_sigma_ref, dim=1)



    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret
        # import pdb; pdb.set_trace()
        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features,
                                "... (joints xyz) -> ... joints xyz",
                                xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]],
                             rotations)
        poses = torch.stack(
            (poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local,
                                      rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat(
            (trajectory[..., :, [0]], root_y[..., None], trajectory[..., :,
                                                                    [1]]),
            dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]
        # import pdb; pdb.set_t
        if self.force_in_meter:
            # different jointstypes have different scale factors
            if self.jointstype == 'mmm':
                factor = 1000.0
            elif self.jointstype in ['humanml3d', 'motionx']:
                factor = 1000.0 * 0.75 / 480.0

            # return results in meters
            return (remove_padding(poses / factor, lengths),   # torch.Size([32, 196, 52, 3])
                    remove_padding(poses_local / factor, lengths), #torch.Size([32, 196, 51, 3])
                    remove_padding(root / factor, lengths),
                    remove_padding(trajectory / factor, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local,
                                   lengths), remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
