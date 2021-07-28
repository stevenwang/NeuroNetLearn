#########################################################################
## genlasso solver 
## use JIT from numba to speed up the general lasso solver
## the solver is based on the smoothing gradient descent algorithm (SPG)
##
#########################################################################

from numba import jit
import numpy as np



@jit(nopython=True)
def funS( x):
    return np.sign(x)* np.minimum( np.abs(x), 1)

@jit(nopython=True)
def softThresh( x, lam):
    return np.sign(x)* np.maximum( np.abs(x) - lam, 0)


@jit(nopython=True)
def spg_genlasso_LS( R,  XX, RR, XY, Rt, Xt, 
                    lam1,  beta_init,  mu, max_iter , tol , L ,msg):

    w_pre = beta_init
    beta_pre = beta_init
    theta_pre = 1.0
    theta = 1.0
    conv = False 

    a_star = np.zeros( R.shape[0],)
    dh = np.zeros( XX.shape[0], )

    v = np.zeros(  len(beta_init), )
    beta = np.zeros(  len(beta_init), )
    diff = 1.0

    ITER = 1
    

    while not conv:
        a_star= funS( R.dot(w_pre) / mu )        
        dh = XX.dot( w_pre ) - XY + Rt.dot(  a_star )        
        v = w_pre - 1/L * dh 
        beta = softThresh(v , lam1/L)
        diff = np.max( np.abs(beta - beta_pre)) / np.mean( np.abs(beta_pre) + 1e-8 ) 

        if diff < tol or ITER > max_iter:
            conv = True 

        theta = 2.0/( ITER + 3.0) 

        w_pre = beta + (1.0-theta_pre)/theta_pre * theta *(beta- beta_pre) 
        beta_pre = beta   
        theta_pre = theta 
        ITER = ITER + 1.0

        if ITER % 100 == 0 and msg:
            print( "iter:", ITER, "diff:", diff,"\n")
            
    return beta, ITER, ITER < max_iter
    


