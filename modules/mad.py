#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

from torch.nn import Module
from modules._masker import Masker, MaskerCNN
from modules._fnn_denoiser import FNNDenoiser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['MaD',  'MaDConv']


class MaD(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim,
                 context_length, original_input_dim):
        super(MaD, self).__init__()

        self.masker = Masker(
            rnn_enc_input_dim=rnn_enc_input_dim,
            rnn_dec_input_dim=rnn_dec_input_dim,
            context_length=context_length,
            original_input_dim=original_input_dim
        )

        self.denoiser = FNNDenoiser(
            input_dim=original_input_dim
        )

        self.output = namedtuple(
            'mad_output',
            ['v_j_filt_prime', 'v_j_filt', 'h_enc', 'h_dec']
        )

    def forward(self, x):
        """The forward pass of the MaD.

        :param x: The input to the MaD.
        :type x: torch.Tensor
        :return: The output of the MaD. The\
                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
                   - `h_enc`, the output of the RNN encoder
                   - `h_dec`, the output of the RNN decoder
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor\
                torch.Tensor, torch.Tensor]
        """
        # Masker pass
        m_out = self.masker(x)

        # Denoiser pass
        v_j_filt = self.denoiser(m_out.v_j_filt_prime)

        return self.output(
            m_out.v_j_filt_prime,
            v_j_filt,
            m_out.h_enc,
            m_out.h_dec
        )


class MaDConv(Module):

    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,
                 cnn_dropout, rnn_dec_input_dim,
                 context_length, original_input_dim, latent_n=3):
        super(MaDConv, self).__init__()

        self.masker = MaskerCNN(
            cnn_channels = cnn_channels, inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding, cnn_dropout=cnn_dropout,
            rnn_dec_input_dim=rnn_dec_input_dim,
            context_length=context_length,
            original_input_dim=original_input_dim
            latent_n=latent_n
        )

        self.denoiser = FNNDenoiser(
            input_dim=original_input_dim
        )
        self.output = namedtuple(
            'mad_output',
            ['v_j_filt_prime', 'v_j_filt', 'h_enc', 'h_dec']
        )

    def forward(self, x):
        """The forward pass of the MaD.

        :param x: The input to the MaD.
        :type x: torch.Tensor
        :return: The output of the MaD. The\
                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
                   - `h_enc`, the output of the RNN encoder
                   - `h_dec`, the output of the RNN decoder
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor\
                torch.Tensor, torch.Tensor]
        """
        # Masker pass
        m_out = self.masker(x)

        # Denoiser pass
        v_j_filt = self.denoiser(m_out.v_j_filt_prime)

        return self.output(
            m_out.v_j_filt_prime,
            v_j_filt,
            m_out.o_enc,
            m_out.o_dec
        )


# EOF
