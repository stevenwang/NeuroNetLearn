# utility functions 
import numpy as np
from itertools import combinations

# import general lasso solver 
from spg_genlasso_solver_jit import spg_genlasso_LS

# spg genlasso wrapper
def spg_genlasso_multi_lam( Y, # long vector stack outcomes 
                            X , # block matrix, 
                            W,  #  weight mat 
                lams , # tuning parameters (lam1, lam2) each row
                M, p,   # no. exps, no. processes
                beta_init ,  # initial coef values
                eps, # error of smoothing approxiamtion 
                max_iter, # maximum iterations 
                tol , # tolerance level of convergence
                msg # print convergence iterations 
            ):
    # de-deduplicate lams
    lams = np.unique( lams, axis=0)
    nlams = lams.shape[0]

    EST = np.zeros( ( nlams, p , (p+1)*M ) )

    # organize inputs
    XX = X.T.dot( X)
    R  =  getR( M, p,W)
    XX  = X.T.dot(X) / len(Y)
    RR = R.T.dot(R) 
    XY = X.T.dot(Y) / len(Y)
    Rt = R.T
    Xt = X.T/len(Y)
    D = R.shape[0]
    mu = eps / 2.0 / D 

    L_RR = np.real( np.max( np.linalg.eigvals(RR)[0]/mu ) )
    L_XX = np.real( np.max( np.linalg.eigvals(XX) ) )

    if beta_init is None:
        beta_init = np.zeros( M*(p+1 ) )
    
    # indicate l1 penalty position
    intercept = np.tile( np.concatenate( [np.zeros(1),  np.ones(p) ] ) ,  M )
    
    for i in range(p):
        for j in range(nlams):
            lam1, lam2= lams[j]
            L = np.real((lam2**2)* L_RR + L_XX )
            EST[j,i,], _ , _ =spg_genlasso_LS( lam2*R,  XX, RR*(lam2 **2), XY[:,i], Rt*lam2, Xt, \
                    lam1*intercept*L, beta_init,  mu, max_iter , tol ,  L ,False)
            
            
    return EST

# select optimum tuning parameters
from getBIC import getBIC

def optSelect( Y, # dict, outcome
               X, # dict, design mat
               EST # coef est value
             ):
    
    nlams = EST.shape[0]
    p = Y[0].shape[1]
    M = len(Y)

    eBIC = np.zeros( nlams)
    beta_idx = np.arange( (p+1)*M )% (p+1) != 0
    u_idx = np.arange( (p+1)*M )% (p+1) == 0

    for i in range(nlams):
        Beta_i = EST[i][ :, beta_idx]
        Beta_i = np.array( [ Beta_i[: ,  (m*p) : (m*p+p) ] for m in range(M) ] )
        U_i = EST[i][:, u_idx].T

        eBIC[i] = getBIC(Y=Y , X=X, U=U_i, Beta= Beta_i, r= 0.5)

    idx_opt = eBIC.argmin() # idx for best (minimum) eBIC
    #print( 'opt lam', lams[idx_opt] )
    Beta_opt = EST[idx_opt][ :, beta_idx]
    Beta_opt = np.array( [ Beta_opt[: ,  (m*p) : (m*p+p) ] for m in range(M) ] )
    U_opt = EST[idx_opt][:, u_idx].T
    
    return U_opt, Beta_opt , idx_opt



def net_est_joint( Y, # outcome, dict, each  T x p array
             X=None, # design matrix , dict, optional, if not given, generate in the loop
             method = 'joint',
             W = None, # weight matrix, optional, if not given, generate in the loop
             lams = None,
             beta_init = None ,  # initial coef values
             eps =1e-4, # error of smoothing approxiamtion 
             max_iter=1e4, # maximum iterations 
             tol =1e-5, # tolerance level of convergence
             msg = False
    ):
    
    # prepare data into matrix format
    M = len(Y)
    p = Y[0].shape[1]
    T = [len(Ym) for Ym in Y.values() ]
 
    # check inputs
    if X is None:
        X = getX( Y , rho=2, rate= 1, lag= 1) 
        
    if W is None:
        if method == 'joint':
            # joint est with informative cross-correlated based weights 
            W = getW(Y, lag= 1, alpha =0.02)
        if method == 'uniform':
            W = normalizeW (  np.ones( (M,M)) )
        if method == 'separate':
            W = np.zeros( (M,M ) )
            
    # reshape input outcome and design matrix
    Ymat = np.vstack( [ Y[m] for m in range(M)]) # reshape into long vector
    Xmat = scipy.linalg.block_diag( *X.values() ) # reshape into diag block mat
    
    
        # auto tuning selection
    if lams is None:
        lam =  1.0/ np.sum(T)/np.sqrt(M) * np.linspace( 0, 2, 5)
        rho =  np.sqrt(M)  * np.linspace(0, 3, 5 )
        lams = [ [l , l*r ] for l in lam for r in rho ]
        lams = np.array( lams)

    if M==1 or method =='separate':
        lams[:,1] = 0

    lams = np.unique( lams, axis=0)

    # create results container
    BETA_EST = np.zeros([ p, p*M] )
    U_EST = np.zeros( [p,M])
    beta_idx = np.arange( (p+1)*M )% (p+1) != 0
    u_idx = np.arange( (p+1)*M )% (p+1) == 0

    EST  = spg_genlasso_multi_lam( Ymat, Xmat, W, lams, M, p, \
                                   beta_init, eps, max_iter, tol, False)

    U_opt, Beta_opt, idx_opt = optSelect( Y, X, EST)


    return {'BETA_EST': Beta_opt  , 'U_EST': U_opt , \
       'lam_opt':  lams[idx_opt], 'lams': lams,  'EST': EST}



# interface used to handle data inputs
import scipy.linalg

def net_est( Y, # outcome, dict, each  T x p array
             X=None, # design matrix , dict, optional, if not given, generate in the loop
             method = 'joint',
             W = None, # weight matrix, optional, if not given, generate in the loop
             lams = None,
             beta_init = None ,  # initial coef values
             eps =1e-4, # error of smoothing approxiamtion 
             max_iter=1e4, # maximum iterations 
             tol =1e-5, # tolerance level of convergence
             msg = False
    ):

    if len(Y)==1:
        method = 'separate'

    if method == 'separate' :
        M = len(Y)
        p = Y[0].shape[1]

        BETA_EST = np.zeros( ( M, p ,p ) )
        U_EST = np.zeros( (M, p) )
        lam_opt = np.zeros( M )
        EST =  []
        lams_all = []

        for m in range( M ):
            ESTm = net_est_joint( Y={0:Y[m]} , 
                            X=X, # design matrix , dict, optional, if not given, generate in the loop
                            method = method,
                            W = W, # weight matrix, optional, if not given, generate in the loop
                            lams = lams,
                            beta_init = beta_init ,  # initial coef values
                            eps =eps, # error of smoothing approxiamtion 
                            max_iter=max_iter, # maximum iterations 
                            tol =tol, # tolerance level of convergence
                            msg = msg  )

            BETA_EST[m] = ESTm['BETA_EST']
            U_EST[m] = ESTm['U_EST']
            lam_opt[m] = ESTm['lam_opt'][0]
            EST.append ( ESTm['EST'] )
            lams_all.append( ESTm['lams'] )

        EST = np.array(EST)
        lams_all = np.array(lams_all)

        return {'BETA_EST': BETA_EST  , 'U_EST': U_EST , \
               'lam_opt':  lam_opt, 'lams': lams_all,  'EST': EST }

    else:
        return net_est_joint( Y, # outcome, dict, each  T x p array
             X=X, # design matrix , dict, optional, if not given, generate in the loop
             method = method,
             W = W, # weight matrix, optional, if not given, generate in the loop
             lams = lams,
             beta_init = beta_init ,  # initial coef values
             eps =eps, # error of smoothing approxiamtion 
             max_iter=max_iter, # maximum iterations 
             tol =tol, # tolerance level of convergence
             msg = msg 
            )


###############################################
# utility function used in net_est procedure 

def getR(M, p, W=None):
    if M == 1:
        mat = np.ones( [1,1])
    else:
        if W is None:
            W = np.ones( (M,M))
            
        comb = list( combinations( range(M), 2 ) )
        N = len( comb)
        mat = np.zeros( (N, M) )
        
        for k in range(N):
            idx = comb[k]
            w_k = W[ idx[0] , idx[1] ]
            mat[ k][ idx[0]] = 1 * w_k 
            mat[ k][ idx[1]] = - 1 * w_k 

    unit = np.diag( np.append( np.array([0]), np.ones(p) ) )

    return np.kron( mat, unit)


# prepare X based on Y using default transition kernel 

from simu_net import integrated

# prepare design matrix using given transition kernel 

def getX( Y, rho=2, rate=1, lag=1): 
    X = dict( )
    M = len(Y)
    T = [ Ym.shape[0] for Ym in Y.values()]
    p = Y[0].shape[1]
    
    for m in range(M):
        y_m = Y[m]
        x_m = np.zeros( [T[m], p] )
        for t in range(T[m]):
            y_pre = y_m[ :t,: ]
            x_t = np.apply_along_axis( integrated, 0, y_pre, \
            	               rho=rho, rate=rate, lag=lag) 
            x_m[ t ,: ] = x_t
        
        X[m] = np.hstack( ( np.ones( (T[m],1)), x_m ) )
    
    return X


# calculate similarity weights

# cross covariance
from scipy.stats import pearsonr , norm

def getCrossCov( Y, lag):

    p = Y.shape[1]
    N = Y.shape[0]
    Cor_mat = np.zeros( [p,p] )
    
    for i in range(p):
        for j in range(p):
            Cor_mat[i,j] , _ =  pearsonr( Y[ lag:N, i], Y[ 0:(N-lag) , j ])
            
    return Cor_mat



def getEmpiricalCorMat( Y , lag , alpha ):
        
    M = len(Y)
    T = [ Ym.shape[0] for Ym in Y.values()]
    p = Y[0].shape[1]

    Connect_Mat = np.zeros( [M,p,p])
    pval_Mat = np.zeros( [M,p,p])

    for m in range(M):
        rho_mat = getCrossCov( Y[m] , lag )
        # fisher transformation to get p-values for coef
        z_mat = 0.5 * (np.log(rho_mat+1) - np.log(1-rho_mat) )
        pval_Mat[m] = 2*(1-norm.cdf( abs(z_mat), scale=1/np.sqrt(T[m]-3) )  )
        Connect_Mat[m]  = 1.0*( pval_Mat[m]  <= alpha )   

    return Connect_Mat , pval_Mat

# get common edge 

def getComEdge( A, B):
    A[ np.isnan(A)] = 0
    B[ np.isnan(B)] = 0
    
    return np.sum( np.sign( np.abs(A)) * np.sign( np.abs(B))  )

# normalize weight
def normalizeW( W):
    tot = np.sum( np.triu(W, k=1) )
    if tot !=0:
        W = W / tot 
   
    return W

# getW

def getW( Y, lag= 1, alpha =0.02 ):

    Mat, _ = getEmpiricalCorMat( Y , lag, alpha)
    
    M = Mat.shape[0]
    p = Mat.shape[1]
    W = np.zeros( [ M,M])

    for i in range(M):
        for j in range(M):
            if i < j:
                W[i,j] = getComEdge( Mat[i] , Mat[j])
                W[j,i] = W[i,j]
    

    return normalizeW(W)





###