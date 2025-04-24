from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric

from .utils import *


# motion reconstruction metric
class MRMetrics_body_hand(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if jointstype not in ["mmm", "humanml3d", "motionx", "motionx_v26"]:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter
        # import pdb; pdb.set_trace()
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("MPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")


        self.add_state("MPJPE_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")


        self.add_state("MPJPE_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")


        self.add_state("PAMPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("PAMPJPE_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("PAMPJPE_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")


        self.add_state("ACCEL",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("ACCEL_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("ACCEL_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        # todo
        # self.add_state("ROOT", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        self.MR_metrics = ["MPJPE", "MPJPE_body", "MPJPE_hand", "PAMPJPE", "PAMPJPE_body", "PAMPJPE_hand", "ACCEL", "ACCEL_body", "ACCEL_hand"]

        # All metric
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):
        if self.force_in_meter:
            # different jointstypes have different scale factors
            # if self.jointstype == 'mmm':
            #     factor = 1000.0
            # elif self.jointstype == 'humanml3d':
            #     factor = 1000.0 * 0.75 / 480
            factor = 1000.0
        else:
            factor = 1.0
        
        count = self.count
        count_seq = self.count_seq
        mr_metrics = {}
        mr_metrics["MPJPE"] = self.MPJPE / count * factor
        mr_metrics["MPJPE_body"] = self.MPJPE_body / count * factor
        mr_metrics["MPJPE_hand"] = self.MPJPE_hand / count * factor


        mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
        mr_metrics["PAMPJPE_body"] = self.PAMPJPE_body / count * factor
        mr_metrics["PAMPJPE_hand"] = self.PAMPJPE_hand / count * factor
        # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        # n-2 for each sequences
        mr_metrics["ACCEL"] = self.ACCEL / (count - 2 * count_seq) * factor
        mr_metrics["ACCEL_body"] = self.ACCEL_body / (count - 2 * count_seq) * factor
        mr_metrics["ACCEL_hand"] = self.ACCEL_hand / (count - 2 * count_seq) * factor
        return mr_metrics

    def update(self, joints_rst: Tensor, joints_ref: Tensor,
               lengths: List[int], name=None):
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)
        # import pdb; pdb.set_trace()


        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # avoid cuda error of DDP in pampjpe
        rst = joints_rst.detach().cpu()
        ref = joints_ref.detach().cpu()
        # import pdb; pdb.set_trace()
        # align root joints index
        if self.align_root and self.jointstype in ['mmm', 'humanml3d', 'motionx', 'motionx_v26']:
            align_inds = [0]
            align_inds_left_wrist = [20]
            align_inds_right_wrist = [21]
        else:
            align_inds = None

        for i in range(len(lengths)):
            all_mpjpe = torch.sum(calc_mpjpe(rst[i], ref[i], align_inds=align_inds))
            self.MPJPE += all_mpjpe
                
            # with open("/comp_robot/lushunlin/motion-latent-diffusion/sort_file/vq_all.txt", "a") as f:
            #     f.write(name[0] + '\n')
            #     f.write(str(float(all_mpjpe.detach().cpu().numpy())) + '\n')
            
            body_mpjpe = torch.sum(calc_mpjpe(rst[i][..., :22, :], ref[i][..., :22, :], align_inds=align_inds))
            self.MPJPE_body += body_mpjpe
                
            # with open("/comp_robot/lushunlin/motion-latent-diffusion/sort_file/vq_body.txt", "a") as f:
            #     f.write(name[0] + '\n')
            #     f.write(str(float(body_mpjpe.detach().cpu().numpy())) + '\n')

            hand_mpjpe = torch.sum(calc_mpjpe_hand(rst[i], ref[i], align_inds=[align_inds_left_wrist, align_inds_right_wrist]))
            self.MPJPE_hand += hand_mpjpe
                
            # with open("/comp_robot/lushunlin/motion-latent-diffusion/sort_file/vq_hand.txt", "a") as f:
            #     f.write(name[0] + '\n')
            #     f.write(str(float(hand_mpjpe.detach().cpu().numpy())) + '\n')

            self.PAMPJPE += torch.sum(calc_pampjpe(rst[i], ref[i]))
            self.PAMPJPE_body += torch.sum(calc_pampjpe(rst[i][..., :22, :], ref[i][..., :22, :]))
            self.PAMPJPE_hand += torch.sum(calc_pampjpe(rst[i][..., 22:, :], ref[i][..., 22:, :]))

            self.ACCEL += torch.sum(calc_accel(rst[i], ref[i]))
            self.ACCEL_body += torch.sum(calc_accel(rst[i][..., :22, :], ref[i][..., :22, :]))
            self.ACCEL_hand += torch.sum(calc_accel(rst[i][..., 22:, :], ref[i][..., 22:, :]))
