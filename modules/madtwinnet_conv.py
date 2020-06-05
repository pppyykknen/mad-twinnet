#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The MaD TwinNet system.
"""

from collections import namedtuple

from torch.nn import Module

from modules.mad import MaDConv

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
#__all__ = ['MaDTwinNet']


class MaDTwinNet_conv(Module):

    def __init__(self,cnn_channels, inner_kernel_size, inner_padding,
                 cnn_dropout,
                 original_input_dim, context_length, latent_n=3, residual=False):
        """The MaD TwinNet as a module.

        This class implements the MaD TwinNet as a module\
        and it is based on the separate modules of MaD and\
        TwinNet.
        :param cnn_channels: The amount of CNN channels used in the blocks
        :type cnn_channels: int
        :param inner_kernel_size: Size of the kernel used for the inner convolution
        :type inner_kernel_size int
        :param inner_padding: Padding size for the inner convolution
        :type inner_padding: int
        :param cnn_dropout: Dropout rate for the convolutions
        :type cnn_dropout: float
        :param rnn_dec_input_dim: The input dimensionality of\
                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        :param context_length: The amount of time frames used as\
                               context.
        :type context_length: int
        """
        super(MaDTwinNet_conv, self).__init__()

        self.mad = MaDConv(
            cnn_channels = cnn_channels,
            inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding,
            cnn_dropout=cnn_dropout,
            context_length=context_length,
            original_input_dim=original_input_dim,
            latent_n=latent_n,
            residual=residual
        )


        self.output = namedtuple(
            'mad_twin_net_output',
            [
                'v_j_filt_prime',
                'v_j_filt'

            ]
        )

    def forward(self, x):
        """The forward pass of the MaD TwinNet.

        :param x: The input to the MaD TwinNet.
        :type x: torch.Tensor
        :return: The output of the MaD TwinNet. The\
                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser

        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor]
        """
        # Masker pass
        mad_out = self.mad(x)


        return self.output(
            mad_out.v_j_filt_prime,
            mad_out.v_j_filt)
          

# EOF
