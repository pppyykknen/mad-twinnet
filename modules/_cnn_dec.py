#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN dec of the Masker.
"""

import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm2d, MaxPool2d, Dropout2d
from modules.depthwise_separable_conv_block import DepthWiseSeparableConvBlock

__author__ = ['Pyry Pyykkönen -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['CNNDec']


class CNNDec(Module):
    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,cnn_dropout, pool_size=5):
        """The CNN dec of the Masker.

        :param cnn_channels: The amount of CNN channels used in the blocks
        :type cnn_channels: int
        :param inner_kernel_size: Size of the kernel used for the inner convolution
        :type inner_kernel_size int
        :param inner_padding: Padding size for the inner convolution
        :type inner_padding: int
        :param cnn_dropout: Dropout rate for the convolutions
        :type cnn_dropout: float
        :param pool_size: Amount of features pooled in MaxPool2d
        :type pool_size: int
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

        :param h_enc: The output of the CNN encoder.
        :type h_enc: torch.Tensor
        :return: The output of the CNN dec (h_j_dec).
        :rtype: torch.Tensor
        """

        return  self.layer_1(x)

# EOF
