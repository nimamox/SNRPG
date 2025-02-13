import numpy as np

def ISI_encoding(X_i, N=3):

    N_neurons = N # number of ISI encoding neurons
    N_spikes = 2 ** (N_neurons - 1) # number of spikes 

    C  = .5
    V_thresh = -0.65
    beta = 3

    D = np.zeros(8*(2*(2**(N_neurons-1)-1)-1)+1)
    for k in range(1, 2**(N_neurons-1)):
        # case 2*in-1 | 2^(N-1)-1 
        D[2*k-1] = X_i * (1 / beta ** (N_neurons - 1))

        # case 2*(2*in-1) | 2^(N-1)-2
        D[2*(2*k-1)] = X_i * ( (1/(beta**(N_neurons-2))) - (1/(beta**(N_neurons-1))) )
        D[2**(N_neurons-1)-2] = X_i * ( (1/(beta**(N_neurons-2))) - (1/(beta**(N_neurons-1))) )

        # case 4*(2*in-1) | 2^(N-1)-4
        D[4*(2*k-1)] = X_i * ( (1/(beta**(N_neurons-3))) - (1/(beta**(N_neurons-2))) 
                           - (1/(beta**(N_neurons-1))) );
        # case 8*(2*in-1) | 2^(N-1)-8
        D[8*(2*k-1)] = X_i * ( (1/(beta**(N_neurons-4))) - (1/(beta**(N_neurons-3)))
                           - (1/(beta**(N_neurons-2))) - (1/(beta**(N_neurons-1))) );

        if N_neurons > 1:
            D[2**(N_neurons-1)-1] = X_i* (1/(beta**(N_neurons-1)))
        if N_neurons > 2:
            D[2**(N_neurons-1)-2] = X_i * ( (1/(beta**(N_neurons-2))) - (1/(beta**(N_neurons-1))) )
        if N_neurons > 3:
            D[2**(N_neurons-1)-4] = X_i * ((1/(beta**(N_neurons-3))) - (1/(beta**(N_neurons-2)))
                                     - (1/(beta**(N_neurons-1))))
        if N_neurons > 4:
            D[2**(N_neurons-1)-8] = X_i * ( (1/(beta**(N_neurons-4))) - (1/(beta**(N_neurons-3)))
                                      - (1/(beta**(N_neurons-2))) - (1/(beta**(N_neurons-1))))
        # case 2^(N-2)
        cte2 = 1/beta
        for j in range(2, N_neurons):
            cte2 = cte2 - 1/(beta**(j))
        if N_neurons > 1:
            D[2**(N_neurons-2)]= X_i * cte2;
    return D[1:2**(N_neurons-1)]