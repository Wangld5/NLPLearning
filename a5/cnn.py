#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """docstring for CNN"""

    def __init__(self, embed_size=50, m_word=21, k=5, f=None):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels=f, kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, X_reshape: torch.Tensor)->torch.Tensor:
        x_conv = self.conv1d(X_reshape)
        x_convout = self.maxpool(F.relu(x_conv))
        return torch.squeeze(x_convout, -1)

# END YOUR CODE
