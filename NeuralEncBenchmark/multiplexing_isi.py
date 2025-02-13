import numpy as np
from .isi import ISI_encoding

def multiplexing_encoding_ISI_phase(X, N_ISI=3, grouping_size=None, smo_freq=200, TMAX=120, 
                                    max_num_spike=None, return_ISI=False, x_max=255, x_offset=0, 
                                    sorted_spikes=True, chunks=1, with_reshape=True, inverse=False):
    # For very large X matrices increase `chunks` to split the alignment work into `chunks` portions,
    # otherwise it takes too much RAM and might crash!
    if X.ndim != 2:
        raise Exception('The stimuli must be a 2D array')
    if grouping_size is None:
        grouping_size = X.shape[1]
    X = (X * x_max) + x_offset
    unique_vals, inv = np.unique(X, return_inverse = True)
    ISI_cache = {}
    for vv in unique_vals.tolist():
        ISI_cache[vv] = ISI_encoding(vv, N_ISI)
    SMO_interval = 1000 / smo_freq

    D_ISI = np.array([ISI_cache[xx] for xx in unique_vals])[inv].reshape([X.shape[0], X.shape[1], -1])
    max_T = np.cumsum(ISI_encoding(x_offset+x_max, N_ISI))[-1]
    
    TT = np.cumsum(D_ISI, axis=2)
    
    if inverse:
        TT = max_T - TT

    TT = np.clip(TT * TMAX / max_T, 0, TMAX-SMO_interval)

    if max_num_spike is not None:
        max_num_spike = min(max_num_spike, TT.shape[1])
        TT = TT[:,:,:max_num_spike]
    else:
        max_num_spike = TT.shape[2]

    TT_result = np.zeros_like(TT)
    tmax = np.arange(0, TMAX+1, SMO_interval)
    Tmax = np.zeros([grouping_size, tmax.shape[0]])
    for j in range(grouping_size):
        # SMO has a frequency of smo_freq Hz
        Tmax[j,:] = np.clip(tmax + (SMO_interval*j/grouping_size), 0, TMAX)

    # Alignment
    for j in range(grouping_size):
        for k in range(chunks):
            arr = (Tmax[j,:].reshape(1,-1) - TT[k::chunks,j::grouping_size,:].reshape(-1,1))
            minIdx = np.where(arr <= 0, arr, -np.inf).argmax(axis=1)
            TT_result[k::chunks,j::grouping_size,:] = Tmax[j, minIdx + 1].reshape(-1, int(TT.shape[1]/grouping_size), max_num_spike)

    if with_reshape:
        TT_result = TT_result.reshape(X.shape[0], -1, grouping_size*max_num_spike)
        if sorted_spikes:
            TT_result = np.sort(TT_result, axis=2)
    if return_ISI:
        return TT_result, TT
    else:
        return TT_result