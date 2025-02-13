import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from .torch_device import device, dtype

def sparse_generator(d, shuffle=True):
   firing_times = d['firing_times']
   nb_units = d['nb_units']
   sample_index = d['sample_index']
   number_of_batches = d['number_of_batches']
   batch_size = d['batch_size']
   nb_steps = d['nb_steps']
   labels_ = d['labels_']

   unit_numbers = np.arange(nb_units)

   if shuffle:
      np.random.shuffle(sample_index)

   counter = 0
   while counter<number_of_batches:
      batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

      coo = [ [] for i in range(3) ]
      for bc, idx in enumerate(batch_index):
         c = np.logical_and(0<=firing_times[idx], firing_times[idx]<nb_steps)
         for cc in range(c.shape[1]):
            # import pdb; pdb.set_trace()
            times, units = firing_times[idx][c[:,cc], cc], unit_numbers[c[:,cc]]
            # print('*' * 20)
            # print('bc', bc)
            # print(times)
            # print(units)
            # print('*' * 20)

            batch = [bc]*len(times) #[bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

      # i = torch.LongTensor(coo).to(device) 
      i = torch.LongTensor(coo).to(device) # Remove spikes that overlay on eachother 
      v = torch.FloatTensor(np.ones(i.shape[1])).to(device)

      X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
      y_batch = torch.tensor(labels_[batch_index], device=device)
      # return X_batch
      yield X_batch.to(device=device), y_batch.to(device=device)

      counter += 1