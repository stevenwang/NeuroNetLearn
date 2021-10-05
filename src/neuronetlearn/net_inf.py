# network inference 
# based on decorrelated score test
import numpy as np

from sklearn import linear_model
from scipy.stats import norm
from .net_est_auto import getX, net_est

# lasso/l1 penalized regression, wrapper based on sklearn
def lasso( Y, X , cv= 10, random_state= 0):
    clf = linear_model.LassoCV(cv=cv, random_state=random_state, fit_intercept=True )
    clf.fit( X  ,Y)
    return np.hstack( [ clf.intercept_, clf.coef_ ] ) 
    

def getScore(Y, # required, outcome, dict, for all experiments
             X  , # required design matrix
	         COEF=None,  # 3D-array,(M,p,p+1), if not given, est. using lasso 
	         msg= False # print progress, default false, no print
            ):

    """
    calculate de-correlated score statistics
    (ref: Wang et al, 2020: https://arxiv.org/abs/2007.07448) 

    @param Y: dictionary of spiking/response matrix for each condition 
    @param X: dictionary of design matrix for each condition 
    @param COEF: optional, externally estimated model coefficient array of shape (M,p,p+1) for M experiment-network of size p ; 
    if not given, it is estimated via joint estimation 
    @param msg: if True, print learning progress

    @return matrices of test statistic, p-values, lower and upper bound of the estimation 
    """


    # setup 
    M = len(Y)
    p = Y[0].shape[1]
    T = [ len(Ym) for Ym in Y.values()]
    NT = np.sum(T)

    V = np.zeros( (M, p,p) )
    UCI = np.zeros( (M, p,p) )
    LCI = np.zeros( (M, p,p) )
    PVAL = np.zeros( (M, p,p) )

    for m in range(M):
        Xm = X[m]
        Ym = Y[m]
        Tm = T[m]

        for i in range(p):
            if msg:
                print('m:',m, '...i:', i)
            if COEF is None:
                coef = lasso( Ym[:,i] , Xm[:, 1:]  )
            else:
                coef = COEF[m][i] 

            Yhat =  np.maximum( np.minimum(  Xm.dot( coef ) , 1-1e-5 ), 1e-5 )

            for j in range(p):
                # get de-correlated column
                idx =  np.arange(1, j+1, 1 ).tolist() + np.arange( (j+2), (p+1) ).tolist() 
                w_lasso = lasso( Xm[:,j+1], Xm[:, idx]  )
                Xj =  Xm[:,j+1] - Xm[:, [0]+ idx].dot( w_lasso ) 
                eps = Ym[:,i] - (  Xm[:, [0]+ idx ].dot( coef[ [0]+idx ] ) )
                Z = - 1.0/ NT * np.sum(  eps * Xj )
                Gamma_j =  1.0/NT * np.sum(Xj * Xj * Yhat*(1-Yhat)  ) 
                # get score
                V[m,i, j] = np.sqrt(NT) * 1/np.sqrt(Gamma_j) * Z
                # get p-value
                PVAL[m,i, j] =  2*( 1- norm.cdf( np.abs(V[ m,i, j] ) )  )

                # construct confidence interval 
                beta_j_est=  np.sum(eps * Xj ) / np.sum(Xj * Xj )
                var_beta_j_est = 1.0/ np.sum(Xj * Xj )  * np.sum(Xj * Xj * Yhat*(1-Yhat)  )  * 1.0 / np.sum(Xj * Xj )

                LCI [m,i,j] =  beta_j_est - 1.96*np.sqrt(var_beta_j_est )  
                UCI [m,i,j] =  beta_j_est + 1.96*np.sqrt(var_beta_j_est )  
            
    
    return V, PVAL,  LCI, UCI
            

# network inference interface


def net_inf( Y, # outcome, dict, for all experiments
             X = None , # design matrix, dict, default None, created using exponential decay
             COEF=None , # 3D-array, (M,p,p+1),if not given, est. using lasso
             msg = False
           ):

    """
    Network inference for high-dimensional multivariate point process over multi-experiment networks

    @param Y: dictionary of spiking/response matrix for each condition 
    @param X: dictionary of design matrix for each condition 
    @param COEF: optional, externally estimated model coefficient array of shape (M,p,p+1) for M experiment-network of size p ; 
    if not given, it is estimated via joint estimation 
    @param msg: if True, print learning progress

    @return matrices of test statistic, p-values, lower and upper bound of the estimation 
    """
    
    if X is None:
        X =  getX(Y,rho=2, rate=1, lag=1) # with intercept column

    # default use joint estimation 
    if COEF is None:
    	EST = net_est( Y, X=X, method = 'joint'  )
    	BETA_EST = EST['BETA_EST']
    	U_EST = EST['U_EST']
    	COEF =  np.array( [ np.column_stack( [ U_EST[m] , BETA_EST[m]] )  for m in range( len(Y) ) ] )

    return getScore( Y=Y, X=X, COEF=COEF , msg= msg )







            

