# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class LossDropper(nn.Module):
    def __init__(
        self,
        dropc=0.4,
        min_count=10000,
        recompute=10000,
        verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0
        self.percentile_val = 100000000.
        self.cur_idx = 0

        self.verbose = verbose

        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def forward(self, loss):
        if loss is None:
            return loss

        nonzero_loss = loss[torch.nonzero(loss, as_tuple=True)].detach().cpu().numpy()
        self.last_computed += nonzero_loss.shape[0]
        self.count += nonzero_loss.shape[0]

        if self.count < len(self.vals):
            self.vals[self.count - nonzero_loss.shape[0] : self.count] = nonzero_loss
            self.cur_idx += nonzero_loss.shape[0]

            return (loss < np.inf).type(loss.dtype)
        else:
            loss_slice_start = 0
            while loss_slice_start < nonzero_loss.shape[0]:
                loss_slice_len = min(nonzero_loss.shape[0] - loss_slice_start, len(self.vals) - self.cur_idx)
                loss_slice = nonzero_loss[loss_slice_start : loss_slice_start + loss_slice_len]
                self.vals[self.cur_idx: self.cur_idx + len(loss_slice)] = loss_slice

                loss_slice_start += len(loss_slice)

                self.cur_idx += len(loss_slice)
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0

        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return mask
