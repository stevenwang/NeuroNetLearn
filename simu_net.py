# simulate data 

import numpy as np


def integrated( x , rho, rate, lag):
    T = len(x)
    return rho* np.sum( x*np.exp( 0- np.maximum( np.flip(  np.arange( T ) + lag ) , 0 )*rate  ) )


def simu_net( T, M, p, Beta, U, rho, rate, lag):

    M = Beta.shape[0]
    p = Beta.shape[1]

    X = dict()
    Y = dict()
    
    for m in range(M):

        x_m = np.zeros( [T[m], p] )
        y_m = np.zeros( [T[m], p] )

        for t in range(1, T[m], 1):
            y_pre = y_m[ :t,: ]
            x_t = np.apply_along_axis( integrated, 0, y_pre, rho, rate, lag ) 
            y_t = U[m, : ] +   Beta[m].dot( x_t).reshape(-1)
            y_t = 1.0* ( np.random.rand( p ) < y_t )
            x_m[ t ,: ] = x_t
            y_m[t, :] = y_t

        X[m] = x_m 
        Y[m] = y_m 
    
    return {'Spike': Y, 'Phi': X}

