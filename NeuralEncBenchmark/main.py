import os

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torchvision as tv

import pickle
import random
import time

import NeuralEncBenchmark
from NeuralEncBenchmark.ttfs import TTFS_encoder
from NeuralEncBenchmark.isi import ISI_encoding
from NeuralEncBenchmark.multiplexing_ttfs import multiplexing_encoding_TTFS_phase
from NeuralEncBenchmark.multiplexing_isi import multiplexing_encoding_ISI_phase
from NeuralEncBenchmark.datasets import *

from NeuralEncBenchmark.torch_device import dtype, device
from NeuralEncBenchmark.sparse_data_generator import sparse_generator
from NeuralEncBenchmark.surrogate_encoder import encode_data

from NeuralEncBenchmark.surrogate_model import run_snn
from NeuralEncBenchmark.surrogate_train import init_model, compute_classification_accuracy, train

print(device)

def gen_encoded(x_train, y_train, x_test, y_test, encoder_type, grp_size, div_data, nb_steps, TMAX=100, x_max_ISI=255, x_offset_ISI=0):
    if encoder_type not in ['TTFS', 'ISI', 'Phase+TTFS', 'Phase+ISI']:
        raise Exception('Incorrect encoder type')
    if grp_size:
        nb_unites = int(x_train.shape[1]/grp_size)
    else:
        nb_unites = x_train.shape[1]

    mask = np.ones_like(y_train, dtype=np.bool)
    mask[::div_data] = 0

    trn_d = encode_data(x_train[mask,:].reshape(-1, 784), y_train[mask], 
                        nb_units=nb_unites, 
                        encoder_type=encoder_type, 
                        group_size=grp_size,
                      batch_size=512, 
                  nb_steps=nb_steps, 
                  TMAX=TMAX, 
                  x_max_ISI=x_max_ISI, 
                  x_offset_ISI=x_offset_ISI)

    val_d = encode_data(x_train[~mask,:].reshape(-1, 784), y_train[~mask], 
                        nb_units=nb_unites, 
                        encoder_type=encoder_type, 
                        group_size=grp_size,
                      batch_size=1024, 
                  nb_steps=nb_steps, 
                  TMAX=TMAX, 
                  x_max_ISI=x_max_ISI, 
                  x_offset_ISI=x_offset_ISI)

    test_d = encode_data(x_test, y_test, 
                         nb_units=nb_unites, 
                         encoder_type=encoder_type, 
                         group_size=grp_size,
                       batch_size=1024, 
                  nb_steps=nb_steps, 
                  TMAX=TMAX, 
                  x_max_ISI=x_max_ISI, 
                  x_offset_ISI=x_offset_ISI)
    return trn_d, val_d, test_d

confs = (
    # -------
  # {'gs': 1,  'enc': 'TTFS', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'TTFS', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'TTFS', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'TTFS', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 8,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 8,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 16,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 16,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0003, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 16,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0004, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 16,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 16,  'enc': 'Phase+TTFS', 'nb_steps': 100, 'lr': .0006, 'dataset': 'MNIST', 'seed': 1234},
    # -------
  # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 5},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 10},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 15},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 20},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 30},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 45},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 50},
    # -------
  # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 10},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 10},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 50},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 50},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 150},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 150},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 200},
    # {'gs': 8,  'enc': 'Phase+ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'nb_hidden': 200},
    # -------
  {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0005, 'dataset': 'MNIST', 'seed': 1234, 'reg': 1e-5},
)

epochs = 30
time_step = .001
div_data = 5


for c in confs:
    if c['dataset'] == 'MNIST':
        dataset = load_mnist()
    elif c['dataset'] == 'MNISTSMALL':
        mnist = load_mnist()
        sub_mask = np.zeros_like(mnist['y_train'], dtype=np.bool)
        sub_mask[::10] = True
        dataset = {}
        dataset['x_train'] = mnist['x_train'][sub_mask,:]
        dataset['y_train'] = mnist['y_train'][sub_mask]
        dataset['x_test'] = mnist['x_test']
        dataset['y_test'] = mnist['y_test']
    elif c['dataset'] == 'FMNIST':
        dataset = load_fmnist()
    elif c['dataset'] == 'CIFAR10_gray':
        dataset = load_cifar10_gray()
    else:
        raise Exception('Wrong database name')

    invertdata = c.get('invertdata', False)
    if invertdata:
        print('Invert Dataset')
        dataset['x_train'] = dataset['x_train']*-1+1
        dataset['x_test'] = dataset['x_test']*-1+1

    print('*'* 20)
    nb_inputs  = int(dataset['x_train'].shape[1]/c['gs'])
    if not c.get('nb_hidden', False):
        nb_hidden  = int(25*c['gs'])
    else:
        nb_hidden = c['nb_hidden']
    regularization_f = 0
    if c.get('reg', False):
        print("REGULARIZATION", c['reg'])
        regularization_f = c['reg']
    nb_outputs = 10

    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    random.seed(c['seed'])

    tmx = c.get('tmx', 100)
    misi = c.get('misi', 255)
    oisi = c.get('oisi', 0)
    trn_d, val_d, test_d = gen_encoded(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 
                                       c['enc'], c['gs'], div_data, c['nb_steps'], TMAX=tmx, x_max_ISI=misi, x_offset_ISI=oisi)

    print(c['dataset'])
    print('Encoder: {}, group_size: {}, lr: {}, nb_steps: {}, nb_hidden: {}'.format(c['enc'], c['gs'], c['lr'], c['nb_steps'], nb_hidden))
    if 'ISI' in c['enc']:
        print('tmx: {}, misi: {}, oisi: {}'.format(tmx, misi, oisi))

    # print('CUDA MEM:')
    params, alpha, beta = init_model(nb_inputs, nb_hidden, nb_outputs, time_step ) #TODO: tau_mem and tau_syn
    print("W1:", params[0].shape)
    print("W2:", params[1].shape)
    total_weights = params[0].numel() + params[1].numel()
    print("#weights:", total_weights)

    loss_hist, train_acc, val_acc, w_traj = train(trn_d, val_d, c['nb_steps'], params, alpha, beta, 
                                                  lr=c['lr'], nb_epochs=epochs, return_weights=True, 
                                                  regularization_factor=regularization_f)
    test_acc = compute_classification_accuracy(test_d, c['nb_steps'], params, alpha, beta)
    print("Test accuracy: %.3f" % test_acc)

    if regularization_f:
        rr = '_reg{}.h'.format(regularization_f)
    else:
        rr = ''

    invv = ''
    if invertdata:
        invv = '_invertdata'

    fpath = 'SSG_{}__{}__gs{}_i{}_h{}_tot{}__epochs{}_lr{}_div{}_nbs{}_ts{}__seed{}_tmx{}_misi{}_oisi{}{}{}'.format(
        'MNIST', c['enc'].replace('+', ''), c['gs'], nb_inputs, nb_hidden, total_weights, epochs, 
        str(c['lr']).replace('.', '_'), div_data, c['nb_steps'], time_step, c['seed'], tmx, misi, oisi, rr, invv
    )
    with open('SurrGradResults/' + fpath, 'wb') as fo:
        pickle.dump({
            'train_acc': train_acc,
          'loss_hist': loss_hist,
          'val_acc': val_acc,
          'test_acc': test_acc,
          'w1_shape': list(params[0].shape),
          'w2_shape': list(params[1].shape),
          'numel': params[0].numel() + params[1].numel(),
          'params': params,
          'w_traj': w_traj,
          'ENC': c['enc'],
          'GS': c['gs'],
          'epochs': epochs,
          'LR': c['lr'],
          'div_data': div_data,
          'nb_steps': c['nb_steps'],
          'time_step': time_step,
          'seed': c['seed'],
          'tmx': tmx,
          'misi': misi,
          'oisi': oisi,
          'reg': regularization_f,
          'invertdata': invertdata
          }, fo, protocol=pickle.HIGHEST_PROTOCOL)
        fo.flush()
    print(fpath)