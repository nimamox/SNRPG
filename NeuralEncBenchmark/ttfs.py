import numpy as np

def TTFS_encoder(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    "TTFS (Time to first spike)"
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    T[idx] = tmax
    return T

