### get eBIC 

import scipy 
import numpy as np


def logChoose( p, v):
    if v == 0 or p == v:
        return 0 
    else:
        return np.sum( np.log( np.arange(p)+1) ) -  \
                    np.sum( np.log( np.arange( (p-v) )+1) ) - np.sum( np.log(np.arange(v)+1))
    return 0



def getMSE( X, Y , coef, refit):
    
    if not refit:
        return  np.mean( (Y - X.dot(coef) )**2 )        
    else:
        Xrefit = X[:,coef != 0 ]
        # ls fit 
        coef_refit = np.linalg.inv( Xrefit.T.dot(Xrefit) ).dot( Xrefit.T.dot( Y) )
        return np.mean( (Y - Xrefit.dot(coef_refit) )**2 )       
        
    

def getBIC( Y,  # spike as dict
            X , # design matrix as dict 
            U, # background intensity, (M, p)
            Beta, # connectivity matrix (M, p, p)
            r # r used in eBIC 
          ):

    """
    calculate eBIC (extended Bayesian information criterion) 
    ref: http://www3.stat.sinica.edu.tw/sstest/oldpdf/A22n26.pdf

    @param Y: dictionary of spiking/response matrix for each condition 
    @param X: dictionary of design matrix for each condition 
    @param U: background intensity matrix in shape (M,p)
    @param Beta: connectivity matrix in shape (M, p, p)
    @param r: penalty parameter in eBIC

    @return eBIC value for the given data and model parameters
    """

    M = len(Y)
    p = Y[0].shape[1]
    T = [len(Ym) for Ym in Y.values() ]

    #MSE = np.zeros( (M,p) )
    eBIC = np.zeros( (M,p) )
    for i in range(p):
        for m in range(M):
            coef_m = np.hstack( [ U[m][i] , Beta[m][i, ] ]  ) 
            mse =  getMSE( X=X[m], Y=Y[m][:,i], coef= coef_m, refit=True)
            loss = np.log( mse ) * T[m]
            v = np.sum( Beta[m][i, ]  != 0 )
            bic = loss +  np.log(T[m]) * (v+1)
            eBIC[m,i] =   bic + 2*r*logChoose( p, v )
            #MSE[m,i] = mse
            
    return np.sum(eBIC)


