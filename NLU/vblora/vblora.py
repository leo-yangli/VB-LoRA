"""
This file contains the implementation of VBLoRA and its derived classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np


class VBLoRA:
    def __init__(
        self, num_vectors, vector_length, logits_A_shape, logits_B_shape, topk=2
    ):
        """ """
        super().__init__()
        self.vector_bank = torch.nn.Parameter(torch.zeros(num_vectors, vector_length))
        self.logits_A = nn.Parameter(torch.zeros(*logits_A_shape))
        self.logits_B = nn.Parameter(torch.zeros(*logits_B_shape))
        self.topk = topk

    def _get_low_rank_matrix(self, logits):
        top_k_logits, indices = logits.topk(self.topk, dim=-1)
        topk_weights = F.softmax(top_k_logits, dim=-1)
        return (topk_weights.unsqueeze(-1) * self.vector_bank[indices]).sum(-2)


class VBLinear(nn.Linear, VBLoRA):
    def __init__(
        self,
        in_features,
        out_features,
        vector_length,
        num_vectors,
        rank,
        topk=2,
        fan_in_fan_out=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        assert (
            in_features % vector_length == 0 and out_features % vector_length == 0
        ), f"in_features {in_features} and out_features {out_features} must be divisible by vector_length {vector_length}"
        VBLoRA.__init__(
            self,
            num_vectors,
            vector_length,
            logits_A_shape=(in_features // vector_length, rank, num_vectors),
            logits_B_shape=(out_features // vector_length, rank, num_vectors),
            topk=topk,
        )

        self.fan_in_fan_out = fan_in_fan_out

        assert (
            in_features % vector_length == 0
        ), f"in_features {in_features} must be divisible by vector_length {vector_length}"
        assert (
            out_features % vector_length == 0
        ), f"out_features {out_features} must be divisible by vector_length {vector_length}"

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def forward(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        A = (
            self._get_low_rank_matrix(self.logits_A)
            .transpose(1, 2)
            .reshape(-1, self.logits_A.shape[1])
        )  # batch, rank, vector_length
        B = (
            self._get_low_rank_matrix(self.logits_B)
            .transpose(0, 1)
            .reshape(self.logits_B.shape[1], -1)
        )
        return F.linear(x, T(self.weight), bias=self.bias) + x @ A @ B
