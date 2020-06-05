#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Masker module.
"""

from collections import namedtuple

from torch.nn import Module, ConvTranspose2d, Sequential, Conv2d

from modules import _fnn, _cnn_enc, _cnn_dec

__author__ = 'Pyry Pyykkönen -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['MaskerCNN', 'Residual_latent']

class Residual_latent(Module):
    def __init__(self,cnn_channels, kernel_size, padding, cnn_dropout, latent_pool=1):
        super(Residual_latent, self).__init__()
        # must have pool_size =1 ,otherwise cannot add
        self.layer = _cnn_dec.CNNDec(cnn_channels,
                        kernel_size,
                        padding,
                        cnn_dropout, pool_size=latent_pool)
    def forward(self, x):
        return self.layer(x)+x

class MaskerCNN(Module):

    def __init__(self, cnn_channels, inner_kernel_size, inner_padding,cnn_dropout, context_length, original_input_dim, latent_n=3, residual=False):
        """The Masker module of the MaD TwinNet.

        :param cnn_channels: The amount of CNN channels used in the blocks
        :type cnn_channels: int
        :param inner_kernel_size: Size of the kernel used for the inner convolution
        :type inner_kernel_size int
        :param inner_padding: Padding size for the inner convolution
        :type inner_padding: int
        :param cnn_dropout: Dropout rate for the convolutions
        :type cnn_dropout: float
        :param context_length: The amount of time steps used for\
                               context length.
        :type context_length: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        :param latent_n: Amount of additional layers before decoder.
        :type latent_n: int
        :param residual: Use residual connections between additional layers
        :type residual: bool
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

            latent_pool = 1
            self.latent_layers = latent_n
            latents = []
            for i in range(self.latent_layers):
                latents.append(_cnn_dec.CNNDec(  cnn_channels,
                    1,
                    0,
                    cnn_dropout, pool_size=latent_pool)
                               if not residual else
                    Residual_latent( cnn_channels,
                    1,
                    0,
                    cnn_dropout))
                end_feature_n = (end_feature_n // latent_pool) if not residual else end_feature_n# from depth block pooling  stride
            self.latent_conv = Sequential(*latents)
            up_kernel = 2

            self.upsamp = ConvTranspose2d(cnn_channels, cnn_channels, kernel_size=[1,up_kernel], stride=[1,up_kernel], padding=[0,up_kernel//2+1])
            end_feature_n = (end_feature_n -1)*up_kernel   - 2 * (up_kernel//2+1) + (up_kernel//2 +1 -1)  +1  # from convtrans

        self.cnn_dec = _cnn_dec.CNNDec(
            cnn_channels,
            inner_kernel_size,
            inner_padding,
            cnn_dropout
        )
        end_feature_n = (end_feature_n // 5)  # from depth block stride and channel count

        self.cnn_dec2 = _cnn_dec.CNNDec(
            cnn_channels,
            inner_kernel_size,
            inner_padding,
            cnn_dropout
        )
        end_feature_n = (end_feature_n // 5)  # from depth block stride and channel count



        self.channel_reduce = Conv2d(cnn_channels, 8,1, bias=False) # reduce channels
        end_feature_n = 8*end_feature_n
        self.fnn = _fnn.FNNMasker(
            input_dim= end_feature_n, # if self.use_latent else 5184, # hard-coded for now due to lazyness, fix later
            output_dim=original_input_dim,
            context_length=context_length
        )

        self.output = namedtuple(
            typename='masker_output',
            field_names=['v_j_filt_prime']
        )

    def forward(self, x):
        """Forward pass of the Masker.

        :param x: The input to the Masker.
        :type x: torch.Tensor
        :return: The outputs of the CNN encoder,\
                 CNN decoder, and the FNN.
        :rtype: collections.namedtuple
        """
        o_enc = self.cnn_enc(x[:,:,self.context_length:-self.context_length,:])
        # print(o_enc.size())

        if self.use_latent:
            o_enc = self.latent_conv(o_enc)
            o_enc = self.upsamp(o_enc)

        o_dec = self.cnn_dec(o_enc)
        o_dec = self.cnn_dec2(o_dec)
        o_dec = self.channel_reduce(o_dec)


        o_dec = o_dec.permute(0,2,1,3).contiguous().view(o_dec.size(0), o_dec.size(2), -1)

        return self.output(self.fnn(o_dec, x.squeeze(1)))
# EOF
