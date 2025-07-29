from typing import List
import random
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
import os
from .utils import *



class ACCMetrics(Metric):

    def __init__(self,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "acc"

        # add metrics
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.metrics = []
        # Accuracy
        self.add_state("accuracy",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("gt_accuracy",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.metrics.extend(["accuracy", "gt_accuracy"])

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # Accuracy
        self.accuracy = torch.trace(self.confusion) / torch.sum(self.confusion)
        self.gt_accuracy = torch.trace(self.gt_confusion) / torch.sum(
            self.gt_confusion)

        # cat all embeddings
        all_labels = torch.cat(self.label_embeddings, axis=0)
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0)
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0)
        all_gtmotions2 = all_gtmotions.clone()[
            torch.randperm(all_gtmotions.shape[0]), :]
        genstats = calculate_activation_statistics(all_genmotions)
        gtstats = calculate_activation_statistics(all_gtmotions)
        gtstats2 = calculate_activation_statistics(all_gtmotions2)

        all_labels = all_labels.cpu()

        # calculate diversity and multimodality
        self.Diversity, self.Multimodality = calculate_diversity_multimodality(
            all_genmotions,
            all_labels,
            self.num_labels,
            diversity_times=self.diversity_times,
            multimodality_times=self.multimodality_times)

        self.gt_Diversity, self.gt_Multimodality = calculate_diversity_multimodality(
            all_gtmotions, all_labels, self.num_labels)

        metrics.update(
            {metric: getattr(self, metric)
             for metric in self.metrics})

        # Compute Fid
        metrics["FID"] = calculate_fid(gtstats, genstats)
        metrics["gt_FID"] = calculate_fid(gtstats, gtstats2)

        return {**metrics}

    def update(
        self,
        pred_idx: List, 
        label: List, 
        lengths: List[int]
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        


