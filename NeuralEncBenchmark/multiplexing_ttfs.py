import numpy as np
from .ttfs import TTFS_encoder

def multiplexing_encoding_TTFS_phase(X, tau=20, grouping_size=None, smo_freq=200, TMAX=150, 
                                     return_TTFS=False, sorted_spikes=True):
    if X.ndim != 2:
        raise Exception('The stimuli must be a 2D array')
    if grouping_size is None:
        grouping_size = X.shape[1]

    SMO_interval = 1000 / smo_freq
    TT = np.clip(TTFS_encoder(np.array(X), tau=tau, tmax=TMAX), 0, TMAX-1)
    TT_result = np.zeros_like(TT)
    tmax = np.arange(0, TMAX+1, SMO_interval)
    Tmax = np.zeros([grouping_size, tmax.shape[0]])
    for j in range(grouping_size):
        # SMO has a frequency of smo_freq Hz
        Tmax[j,:] = np.clip(tmax + (1000/smo_freq*j/grouping_size), 0, TMAX)
    # Alignment
    for j in range(grouping_size):
        arr = (Tmax[j,:].reshape(1,-1) - TT[:,j::grouping_size].reshape(-1,1))
        minIdx = np.where(arr <= 0, arr, -np.inf).argmax(axis=1)
        # return arr
        TT_result[:,j::grouping_size] = Tmax[j, minIdx + 1].reshape(X.shape[0], -1)
    TT_result = TT_result.reshape(X.shape[0], -1, grouping_size)
    if sorted_spikes:
        TT_result = np.sort(TT_result, axis=2)
    if return_TTFS:
        return TT_result, TT
    else:
        return TT_result