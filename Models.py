import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import *

class LIFNeuron(nn.Module):
    def __init__(self, dim_in, Rd, Cm, Rs, Vth, V_reset, dt):
        super().__init__()
        # self.batch_size = batch_size
        self.dim_in = dim_in
        self.rd = Rd
        self.cm = Cm
        self.rs = Rs
        self.vth = Vth
        self.v_reset = V_reset
        # self.v = torch.full([self.batch_size, self.dim_in], self.v_reset).to(device)
        self.dt = dt

        self.tau_in = 1/(self.cm*self.rs)
        self.tau_lk = 1/(self.cm)*(1/self.rd + 1/self.rs) 

    @staticmethod
    def soft_spike(x):
        a = 2.0
        return torch.sigmoid_(a*x)

    def spiking(self):
        if self.training == True:
            spike_hard = torch.gt(self.v, self.vth).float()
            spike_soft = self.soft_spike(self.v - self.vth)
            v_hard = self.v_reset*spike_hard + self.v*(1 - spike_hard)
            v_soft = self.v_reset*spike_soft + self.v*(1 - spike_soft)
            self.v = v_soft + (v_hard - v_soft).detach_()
            return spike_soft + (spike_hard - spike_soft).detach_()
        else:
            spike_hard = torch.gt(self.v, self.vth).float()
            self.v = self.v_reset*spike_hard + self.v*(1 - spike_hard)
            return spike_hard


    def forward(self, v_inject):
        'Upgrade membrane potention every time step by differantial equation.'
        # print(v_inject.shape)
        self.v += (self.tau_in*v_inject - self.tau_lk*self.v) * self.dt
        return self.spiking(), self.v

    def reset(self, batch_size):
        'Reset the membrane potential to reset voltage.'
        self.v = torch.full([batch_size, self.dim_in], self.v_reset).to(device)

class MAC_Crossbar(nn.Module):
    def __init__(self, dim_in, dim_out, W_std):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = nn.Parameter(torch.zeros(dim_in, dim_out).to(device))
        torch.nn.init.normal_(self.weight, mean=0.0, std=W_std)

    def forward(self, input_vector):
        output = input_vector.mm(self.weight)
        return output

class Three_Layer_SNN(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear1 = MAC_Crossbar(param['dim_in'], param['dim_h'], param['W_std1'])
        self.neuron1 = LIFNeuron(param['dim_h'], param['Rd'], param['Cm'],
                               param['Rs'], param['Vth'], param['V_reset'], param['dt'])
        self.linear2 = MAC_Crossbar(param['dim_h'], param['dim_out'], param['W_std2'])
        self.neuron2 = LIFNeuron(param['dim_out'], param['Rd'], param['Cm'], 
                               param['Rs'], param['Vth']*20, param['V_reset'], param['dt'])

    def forward(self, input_vector):
        out_vector = self.linear1(input_vector)
        # debug print, very useful to see what happend in every layer
        #print('0', out_vector.max())
        # out_vector = self.BatchNorm1(out_vector)
        #print('1', out_vector.max())
        out_vector, _ = self.neuron1(out_vector)
        #print('2', out_vector.sum(1).max())
        out_vector = self.linear2(out_vector)
        #print('3', out_vector.max())
        # out_vector = self.BatchNorm2(out_vector)
        #print('4', out_vector.max())
        out_vector, out_v = self.neuron2(out_vector)
        #print('5', out_vector.sum(1).max())
        return out_vector, out_v

    def reset_(self, batch_size):
        '''
        Reset all neurons after one forward pass,
        to ensure the independency of every input image.
        '''
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset(batch_size)

class LinReg(nn.Module):
    def __init__(self, inputSize, outputSize):
        nn.Module.__init__(self)
        self.linear = nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out

class MLP(nn.Module):
    def __init__(self, inputSize, outputSize, h=10):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(inputSize, h)
        self.linear2 = nn.Linear(h, outputSize)
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out

def Poisson_encoding(x):
    out_spike = torch.rand_like(x).le(x).float()
    return out_spike

def Poisson_encoder(x, T_sim):
    out_spike = torch.zeros([x.shape[0], x.shape[1], T_sim])
    for t in range(T_sim):
        out_spike[:,:,t] = Poisson_encoding(x)
    return out_spike.to(device)


class InputLayer(nn.Module):
    def __init__(self, N, dim_input, dim_output, weight=100.0):
        super().__init__()

        pre = np.arange(dim_input * N) % dim_input
        post = (
          np.random.permutation(max(dim_input, dim_output) * N)[: dim_input * N]
         % dim_output
      )
        i = torch.LongTensor(np.vstack((pre, post)))
        v = torch.ones(dim_input * N) * weight

        # Save the transpose version of W for sparse-dense multiplication
        self.W_t = torch.sparse.FloatTensor(
          i, v, torch.Size((dim_input, dim_output))
         ).t()

    def forward(self, x):
        return self.W_t.mm(x.t()).t()

    def _apply(self, fn):
        super()._apply(fn)
        self.W_t = fn(self.W_t)
        return self

class ReadoutLayer(nn.Module):
    def __init__(self, reservoir_size, dim_input, dim_output):
        super().__init__()
        self.pre = np.random.permutation(np.arange(reservoir_size))[:dim_input]
        self.post = np.arange(dim_input) % dim_output
        i = torch.LongTensor(np.vstack((self.pre, self.post)))
        v = torch.ones(i.shape[1])
        self.W_t = torch.sparse.FloatTensor(
          i, v, torch.Size((reservoir_size, dim_output))
         ).t()
    def forward(self, x):
        res = self.W_t.mm(x.t()).t()
        res[res > .5] = 1
        return res

    def _apply(self, fn):
        super()._apply(fn)
        self.W_t = fn(self.W_t)
        return self