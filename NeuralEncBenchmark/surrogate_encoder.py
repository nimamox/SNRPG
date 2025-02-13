import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from .torch_device import device, dtype
from .ttfs import TTFS_encoder
from .isi import ISI_encoding
from .multiplexing_ttfs import multiplexing_encoding_TTFS_phase
from .multiplexing_isi import multiplexing_encoding_ISI_phase
import pickle

def encode_data(X, y, batch_size, nb_units, encoder_type = "TTFS", nb_steps=1000, TMAX=100, 
                          ISI_N=3, tau=20, group_size=None, 
                          x_max_ISI=255, x_offset_ISI=0, smo_freq=200, external_ISI_cache=None):
  labels_ = np.array(y, dtype=np.int)
  number_of_batches = len(X)//batch_size
  sample_index = np.arange(len(X))

  # compute discrete firing times
  if encoder_type is None:
    firing_times = X
  elif encoder_type == "TTFS":
    firing_times_TTFS = np.array(TTFS_encoder(X, tau=tau, tmax=TMAX) / (TMAX / nb_steps), dtype=np.int)
    # firing_times_TTFS = np.array(TTFS_encoder(X, tau=tau, tmax=TMAX), dtype=np.int)
    # return firing_times_TTFS
    firing_times = firing_times_TTFS.reshape([X.shape[0], X.shape[1], -1])
  elif encoder_type.startswith("ISI"):
    X = (X * x_max_ISI) + x_offset_ISI
    unique_vals, inv = np.unique(X, return_inverse = True)
    if external_ISI_cache is None:
      ISI_cache = {}
    else:
      ISI_cache = external_ISI_cache
      
    for vv in unique_vals.tolist():
      if vv not in ISI_cache:
        ISI_cache[vv] = ISI_encoding(vv, ISI_N)
    # return ISI_cache

    tmax_ISI = np.sum(ISI_encoding(x_max_ISI+x_offset_ISI, ISI_N))

    firing_times_ISI = np.array([ISI_cache[xx] for xx in unique_vals])[inv].reshape([X.shape[0], X.shape[1], -1])

    # Transform distances to exact times
    firing_times_ISI = np.cumsum(firing_times_ISI, axis=2)
    
    # Modified (inverse) ISI, making it more similar to TTFS
    if encoder_type=='ISI_inverse':
      firing_times_ISI = tmax_ISI - firing_times_ISI
    
    # Scaling firing times 
    firing_times_ISI = firing_times_ISI / tmax_ISI * nb_steps
    
    firing_times_ISI = np.array(firing_times_ISI, dtype=np.int)
    firing_times = firing_times_ISI.reshape([X.shape[0], X.shape[1], -1])
  elif encoder_type == "Phase+TTFS":
    firing_times_Mult_TTFS = np.array(multiplexing_encoding_TTFS_phase(X, grouping_size=group_size, 
                                                  TMAX=TMAX, smo_freq=smo_freq) / (TMAX / nb_steps), dtype=np.int)
    firing_times = firing_times_Mult_TTFS 
  elif encoder_type.startswith("Phase+ISI"):
    inverse = (encoder_type == "Phase+ISI_inverse")
    firing_times_Mult_ISI = np.array(multiplexing_encoding_ISI_phase(X, grouping_size=group_size, x_max=x_max_ISI, chunks=20,
                         x_offset=x_offset_ISI, TMAX=TMAX, inverse=inverse, smo_freq=smo_freq) / (TMAX / nb_steps), dtype=np.int)
    firing_times = firing_times_Mult_ISI 
  else:
    raise Exception
  

  # firing_times = firing_times.reshape([X.shape[0], X.shape[1], -1])
  # print('~'*10, firing_times.shape[1], nb_units)
  if firing_times.shape[1] != nb_units:
    raise Exception('Incorrect nb_units: nb_units={}, firing_time.shape={}'.format(nb_units, firing_times.shape))

  return {'firing_times': firing_times, 
          'nb_units': nb_units,
          'sample_index': sample_index,
          'number_of_batches': number_of_batches,
          'batch_size': batch_size,
          'nb_steps': nb_steps,
          'labels_': labels_
          }
