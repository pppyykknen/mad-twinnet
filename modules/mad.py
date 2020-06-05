#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

from torch.nn import Module
from modules._masker import MaskerCNN
from modules._fnn_denoiser import FNNDenoiser

__author__ = 'Pyry Pyykkönen  -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [ 'MaDConv']


class MaDConv(Module):

    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,
                 cnn_dropout,
                 context_length, original_input_dim, latent_n=3, residual=False):
        super(MaDConv, self).__init__()

        self.masker = MaskerCNN(
            cnn_channels = cnn_channels, inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding, cnn_dropout=cnn_dropout,
            context_length=context_length,
            original_input_dim=original_input_dim,
            latent_n=latent_n,
            residual=residual
        )

        self.denoiser = FNNDenoiser(
            input_dim=original_input_dim
        )
        self.output = namedtuple(
            'mad_output',
            ['v_j_filt_prime', 'v_j_filt']
        )

    def forward(self, x):
        """The forward pass of the MaD.

        :param x: The input to the MaD.
        :type x: torch.Tensor
        :return: The output of the MaD. The\
                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensorr]
        """
        # Masker pass
        m_out = self.masker(x)

        # Denoiser pass
        v_j_filt = self.denoiser(m_out.v_j_filt_prime)

        return self.output(
            m_out.v_j_filt_prime,
            v_j_filt
        )


# EOF
