#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Masker module.
"""

from collections import namedtuple

from torch.nn import Module, ConvTranspose2d, Sequential
from torch import flatten

from modules import _rnn_enc, _rnn_dec, _fnn, _cnn_enc, _cnn_dec
#from torch.nn.modules.flatten import Flatten

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['Masker']


class Masker(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim,
                 context_length, original_input_dim):
        """The Masker module of the MaD TwinNet.

        :param rnn_enc_input_dim: The input dimensionality for\
                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality for\
                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param context_length: The amount of time steps used for\
                               context length.
        :type context_length: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        """
        super(Masker, self).__init__()

        self.rnn_enc = _rnn_enc.RNNEnc(
            input_dim=rnn_enc_input_dim,
            context_length=context_length
        )
        self.rnn_dec = _rnn_dec.RNNDec(
            input_dim=rnn_dec_input_dim
        )

        self.fnn = _fnn.FNNMasker(
            input_dim=rnn_dec_input_dim,
            output_dim=original_input_dim,
            context_length=context_length
        )

        self.output = namedtuple(
            typename='masker_output',
            field_names=['h_enc', 'h_dec', 'v_j_filt_prime']
        )

    def forward(self, x):
        """Forward pass of the Masker.

        :param x: The input to the Masker.
        :type x: torch.Tensor
        :return: The outputs of the RNN encoder,\
                 RNN decoder, and the FNN.
        :rtype: collections.namedtuple
        """
        h_enc = self.rnn_enc(x)
        h_dec = self.rnn_dec(h_enc)
        return self.output(h_enc, h_dec, self.fnn(h_dec, x))

class MaskerCNN(Module):

    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,cnn_dropout, rnn_dec_input_dim,
                 context_length, original_input_dim, latent_n=3):
        """The Masker module of the MaD TwinNet.

        :param rnn_enc_input_dim: The input dimensionality for\
                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality for\
                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param context_length: The amount of time steps used for\
                               context length.
        :type context_length: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        """
        super(MaskerCNN, self).__init__()
        self.context_length = context_length
        end_feature_n = original_input_dim

        self.cnn_enc = _cnn_enc.CNNEnc(
            cnn_channels,
            inner_kernel_size,
            inner_padding,
            cnn_dropout)
        end_feature_n =  (end_feature_n//5)  # from depth block pooling stride

        self.use_latent = True
        if self.use_latent:

            # add conv layers between encoder and decoder, maybe remove pool and add residual connection
            # use depthwise sep conv for this as well

            latent_pool = 1
            self.latent_layers = latent_n
            latents = []
            for i in range(self.latent_layers):
                latents.append(_cnn_dec.CNNDec(  cnn_channels,
                    7,
                    3,
                    cnn_dropout, pool_size=latent_pool))
                end_feature_n = (end_feature_n // latent_pool)  # from depth block pooling  stride
            self.latent_conv = Sequential(*latents)
            up_kernel = 5

            self.upsamp = ConvTranspose2d(cnn_channels, cnn_channels, kernel_size=[1,up_kernel], stride=[1,up_kernel], padding=[0,up_kernel//2+1])
            end_feature_n = (end_feature_n -1)*up_kernel   - 2 * (up_kernel//2+1) + (up_kernel//2 +1 -1)  +1  # from convtrans

        self.cnn_dec = _cnn_dec.CNNDec(
            cnn_channels,
            inner_kernel_size,
            inner_padding,
            cnn_dropout
        )
        end_feature_n = cnn_channels *(end_feature_n // 5)  # from depth block stride and channel count

        self.fnn = _fnn.FNNMasker(
            input_dim= end_feature_n, # if self.use_latent else 5184, # hard-coded for now due to lazyness, fix later
            output_dim=original_input_dim,
            context_length=context_length
        )

        self.output = namedtuple(
            typename='masker_output',
            field_names=['o_enc', 'o_dec', 'v_j_filt_prime']
        )

    def forward(self, x):
        """Forward pass of the Masker.

        :param x: The input to the Masker.
        :type x: torch.Tensor
        :return: The outputs of the RNN encoder,\
                 RNN decoder, and the FNN.
        :rtype: collections.namedtuple
        """
        o_enc = self.cnn_enc(x[:,:,self.context_length:-self.context_length,:])
        if self.use_latent:
            o_enc = self.latent_conv(o_enc)
            o_enc = self.upsamp(o_enc)

        o_dec = self.cnn_dec(o_enc)
        o_dec = o_dec.permute(0,2,1,3).contiguous().view(o_dec.size(0), o_dec.size(2), -1)

        return self.output(o_enc, o_dec, self.fnn(o_dec, x.squeeze(1)))
# EOF
