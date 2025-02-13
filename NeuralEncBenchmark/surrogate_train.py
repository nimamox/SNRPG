from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from .torch_device import device, dtype
from .surrogate_model import run_snn
from .surrogate_encoder import encode_data
from .sparse_data_generator import sparse_generator


def train(encoded_data, val_enc_data, nb_steps, params, alpha, beta, lr=2e-3, nb_epochs=10, return_weights=False, regularization_factor=0):
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []

    train_acc = []
    val_acc = []
    w_traj = []
    
    #nb_hidden = params[1].shape[0]
    #nb_outputs = params[1].shape[1]

    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_generator(encoded_data):
            output, recs = run_snn(x_local.to_dense(), encoded_data['batch_size'], nb_steps, params, alpha, beta)
            _, spks = recs
            m, _ = torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local)
            
            if regularization_factor:
                reg_loss = regularization_factor*torch.sum(spks) # L1 loss on total number of spikes
                reg_loss += regularization_factor*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
                loss_val += reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f, \t %s"%(e+1,mean_loss, datetime.now().strftime("%H:%M:%S")))
        loss_hist.append(mean_loss)

        if return_weights:
            w_traj.append({'w1': params[0].detach().clone(),
                           'w2': params[1].detach().clone()})

        with torch.no_grad():
            # Validation accuracy
            accs = []
            for x_local, y_local in sparse_generator(val_enc_data):
                output,_ = run_snn(x_local.to_dense(), val_enc_data['batch_size'], nb_steps, params, alpha, beta)
                m,_ = torch.max(output,1) # max over time
                _, am = torch.max(m,1)      # argmax over output units
                tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
                accs.append(tmp)
            mean_acc = np.mean(accs)
            val_acc.append(mean_acc)
            print("\tVALID:{:.4f}".format(mean_acc))
    if return_weights:
        return loss_hist, train_acc, val_acc, w_traj
    return loss_hist, train_acc, val_acc
    
def compute_classification_accuracy(encoded_data, nb_steps, params, alpha, beta):
    """ Computes classification accuracy on supplied data in batches. """
    
    nb_hidden = params[1].shape[0]
    nb_outputs = params[1].shape[1]
    
    accs = []
    for x_local, y_local in sparse_generator(encoded_data, shuffle=False):
        output,_ = run_snn(x_local.to_dense(), encoded_data['batch_size'], nb_steps, params, alpha, beta)
        m,_= torch.max(output,1) # max over time
        _, am = torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)

def init_model(nb_inputs, nb_hidden, nb_outputs, time_step, tau_mem = 10e-3, tau_syn = 5e-3):
    alpha   = float(np.exp(-time_step/tau_syn))
    beta    = float(np.exp(-time_step/tau_mem))

    weight_scale = 7*(1.0-beta) # this should give us some spikes to begin with

    w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

    print("init done")
    
    return [w1, w2], alpha, beta
    