#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training process module.
"""
import time
from functools import partial

from torch import cuda, from_numpy
from torch import optim, nn, save
import torch
from helpers import data_feeder, printing
from helpers.settings import debug, hyper_parameters, training_constants, \
    training_output_string, output_states_path, _dataset_parent_dir
from modules.madtwinnet_conv import MaDTwinNet_conv
from objectives import kullback_leibler as kl
import argparse

__author__ = ['Pyry Pyykkönen -- Tampere University', 'Konstantinos Drossos -- TAU', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['training_process']

parser = argparse.ArgumentParser(description='Train CNN madtwin')

parser.add_argument('--layers', dest='layers',
                    help='amount of layers between encoder and decoder',
                    default=5, type=int)
parser.add_argument('--channels', dest='channels',
                    help='Amount of CNN channels',
                    default=64, type=int)

parser.add_argument('--do', dest='do',
                    help='dropout, give in percentages',
                    default=10, type=float)
parser.add_argument('--residual', dest='residual',
                    help='Use residuals in latent layers or no',
                    action='store_true', default=False)

args = parser.parse_args()


def _one_epoch(module, epoch_it, solver, separation_loss,epoch_index, device, max_grad_norm,  lats):
    """One training epoch for MaD TwinNet.

    :param module: The module of MaD TwinNet.
    :type module: torch.nn.Module
    :param epoch_it: The data iterator for the epoch.
    :type epoch_it: callable
    :param solver: The optimizer to be used.
    :type solver: torch.optim.Optimizer
    :param separation_loss: The loss function used for\
                            the source separation.
    :type separation_loss: callable
    :param max_grad_norm: The maximum gradient norm for\
                          gradient norm clipping.
    :type max_grad_norm: float
    :param lats: Amount of layers between encoder and decoder.
    :type lats: ints
    """
    def _training_iteration(_m, _data, _device, _solver, _sep_l, _max_grad_norm):
        """One training iteration for the MaD TwinNet.

        :param _m: The module of MaD TwinNet.
        :type _m: torch.nn.Module
        :param _data: The data
        :type _data: numpy.ndarray
        :param _device: The device to be used.
        :type _device: str
        :param _solver: The optimizer to be used.
        :type _solver: torch.optim.Optimizer
        :param _sep_l: The loss function used for the\
                       source separation.
        :type _sep_l: callable

        :param _max_grad_norm: The maximum gradient norm for\
                               gradient norm clipping.
        :type _max_grad_norm: float
        :return: The losses for the iteration.
        :rtype: list[float]
        """
        
        # Get the data to torch and to the device used
        v_in, v_j = [from_numpy(_d).to(_device) for _d in _data]

        # Forward pass of the module
        output = _m(v_in.unsqueeze(1)) ## add 1 channel dim
        
        # Calculate losses
        l_m = _sep_l(output.v_j_filt_prime, v_j)
        l_d = _sep_l(output.v_j_filt, v_j)


        # Make MaD TwinNet objective
        loss = l_m.add(l_d)

        # Clear previous gradients
        _solver.zero_grad()

        # Backward pass
        loss.backward()

        # Gradient norm clipping
        nn.utils.clip_grad_norm_(_m.parameters(), max_norm=_max_grad_norm, norm_type=2)

        # Optimize
        _solver.step()

        return [l_m.item(), l_d.item()]

    # Log starting time
    time_start = time.time()

    # Do iteration over all batches
    iter_results = [
        _training_iteration(module, data, device, solver, separation_loss, max_grad_norm)
        for data in epoch_it()
    ]

    # Log ending time
    time_end = time.time()
    # Print to stdout
    if epoch_index % 10 == 0 and epoch_index > 1:
        # save model every 10 epochs
        save(module.mad.state_dict(), output_states_path['mad'] + str(lats) +  "latents" + str(args.channels)
             +"features"+str(args.do/100) + "dropout" + ("res" if args.residual else "") + str(epoch_index) + "epochs" + _dataset_parent_dir[7:])
        # print("Validation time!")
        # for _data in valid_it():
        #     v_in, v_j = [from_numpy(_d).to(device) for _d in _data]
        #
        #     # Forward pass of the module
        #     output = module(v_in.unsqueeze(1))  ## add 1 channel dim
        #     l_m = separation_loss(output.v_j_filt_prime, v_j).item()
        #     l_d = separation_loss(output.v_j_filt, v_j).item()
        #     iter_results = [[l_m, l_d]]
        #     printing.print_msg(training_output_string.format(
        #         ep=epoch_index,
        #         t=time_end - time_start,
        #         **{k: v for k, v in zip(['l_m', 'l_d'],
        #                                 [sum(i) / len(iter_results)
        #                                  for i in zip(*iter_results)])
        #            }
        #     ))
        #     break
    else:
        printing.print_msg(training_output_string.format(
            ep=epoch_index,
            t=time_end - time_start,
            **{k: v for k, v in zip(['l_m', 'l_d'],
                                    [sum(i)/len(iter_results)
                                     for i in zip(*iter_results)])
               }
        ))



def training_process():
    """The training process.
    """
    # Check what device we'll be using
    device = 'cuda' if not debug and cuda.is_available() else 'cpu'
    #device = 'cpu'
    # Inform about the device and time and date
    printing.print_intro_messages(device)
    printing.print_msg('Starting training process. Debug mode: {}'.format(debug))
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True
    # Set up MaD TwinNet
    with printing.InformAboutProcess('Setting up MaD TwinNet'):
        mad_twin_net = MaDTwinNet_conv(
            cnn_channels=args.channels,
            inner_kernel_size=1,
            inner_padding=0,
            cnn_dropout=args.do/100,
            original_input_dim=hyper_parameters['original_input_dim'],
            context_length=hyper_parameters['context_length'],
            latent_n=args.layers,
            residual=args.residual
        ).to(device)

    # Get the optimizer
    with printing.InformAboutProcess('Setting up optimizer'):
        optimizer = optim.Adam(
            mad_twin_net.parameters(),
            lr=hyper_parameters['learning_rate']
        )

    # Create the data feeder
    with printing.InformAboutProcess('Initializing data feeder'):
        epoch_it = data_feeder.data_feeder_training(
            window_size=hyper_parameters['window_size'],
            fft_size=hyper_parameters['fft_size'],
            hop_size=hyper_parameters['hop_size'],
            seq_length=hyper_parameters['seq_length'],
            context_length=hyper_parameters['context_length'],
            batch_size=4,#training_constants['batch_size'],
            files_per_pass=training_constants['files_per_pass'],
            debug=debug
        )
        valid_it = []
        # data_feeder.data_feeder_training(
        #     window_size=hyper_parameters['window_size'],
        #     fft_size=hyper_parameters['fft_size'],
        #     hop_size=hyper_parameters['hop_size'],
        #     seq_length=hyper_parameters['seq_length'],
        #     context_length=hyper_parameters['context_length'],
        #     batch_size=1,files_per_pass=2, debug=debug, valid=True)
    lats = args.layers
    print("Using " + str(lats) + " latent layers between encoder and decoder.", flush=True)
    print("Using " + str(args.channels) + " cnn channels.", flush=True)
    print("Using " + str(args.do/100) + " dropout.", flush=True)
    print("Using dataset from folder: " + str(_dataset_parent_dir))
    print(("Using residual connections" if args.residual else "Not using residual connections"))

    print("Number of parameters: ",sum(p.numel() for p in mad_twin_net.parameters() if p.requires_grad))


    printing.print_msg('Training starts', end='\n\n')

    one_epoch = partial(
        _one_epoch, module=mad_twin_net,
        epoch_it=epoch_it, solver=optimizer,
        separation_loss=kl, device=device, max_grad_norm=hyper_parameters['max_grad_norm'],
        lats=lats)

    # Training fixed epoch size
    [one_epoch(epoch_index=e) for e in range(100)]

    printing.print_msg('Training done.', start='\n-- ')

    # Save the model
    with printing.InformAboutProcess('Saving model'):
        save(mad_twin_net.mad.state_dict(),"./outputs/states/" + str(lats) + str(args.channels)+str(int(args.do))+("res" if args.residual else "")+".pt"+ _dataset_parent_dir[7:])

    # Say goodbye!
    printing.print_msg('That\'s all folks!')


def main():
    training_process()


if __name__ == '__main__':
    main()

# EOF
