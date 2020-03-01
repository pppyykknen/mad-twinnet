#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN encoder of the Masker.
"""

import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm2d, MaxPool2d, Dropout2d
from modules._modules.depthwise_separable_conv_block import DepthWiseSeparableConvBlock
__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
##__all__ = ['RNNEnc']


class CNNEnc(Module):
    def __init__(self, cnn_channels, inner_kernel_size, inner_padding, cnn_dropout):
        """The CNN encoder of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param context_length: The context length.
        :type context_length: int
        """
        super(CNNEnc, self).__init__()

        self.layer_1: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=1, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),
            Dropout2d(cnn_dropout))




    def forward(self, x):
        """Forward pass.

        :param v_in: The input to the RNN encoder of the Masker.
        :type v_in: torch.Torch
        :return: The output of the Masker.
        :rtype: torch.Torch
        """
        return  self.layer_1(x)

# EOF
