#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN dec of the Masker.
"""

import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm2d, MaxPool2d, Dropout2d
from modules._modules.depthwise_separable_conv_block import DepthWiseSeparableConvBlock

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
#__all__ = ['RNNDec']


class CNNDec(Module):
    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,cnn_dropout, pool_size=5):
        """The CNN dec of the Masker.

        :param cnn_channels, amount of channels in the depthwise conv block
        :type int

        :param cnn_dropout,
        :type float
        """
        super(CNNDec, self).__init__()
        self.layer_1: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=cnn_channels, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
            Dropout2d(cnn_dropout))

    def forward(self, x):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.Tensor
        :return: The output of the RNN dec (h_j_dec).
        :rtype: torch.Tensor
        """

        return  self.layer_1(x)

# EOF
