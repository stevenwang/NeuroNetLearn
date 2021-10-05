# fast network learn using thresholding method

#  helper function 
import numpy as np 
from .net_est_auto import getCrossCov
from .net_inf import net_inf
from scipy.sparse.csgraph import connected_components

# hybrid thresholding based on cross-covariance matrices

def getConnectedComponentsHybrid( Y  , lag = 1 ,lambda1 =0.2, lambda2=0):
    # ref: Exact Hybrid Covariance Thresholding for Joint Graphical Lasso, Tang et al 2015
    # https://link.springer.com/chapter/10.1007/978-3-319-23525-7_36
    # lag: time-lag in calculating the cross-covariance matrix using time-series/point process data
    # lambda1: threshold for class-specific thresholds 
    # lambda2: global thresholding, default is 0, meaning class-specific thresholding only
    
    """
    get Connected Components using hybrid thresholding based on cross-covariance matrices
    (ref: Tang et al 2015, https://link.springer.com/chapter/10.1007/978-3-319-23525-7_36)

    @param Y: dictionary of spiking/response matrix for each condition 
    @param lag: time lag used to calculate the cross-correlations
    @param lambda1: threshold for class-specific thresholds 
    @param lambda2: global thresholding, default is 0, meaning class-specific thresholding only
    
    @return no. of connected components and connected component sets
    """   

    M = len(Y)
    p = Y[0].shape[1]  
    Cov = np.zeros( ( M , p,p) )
    for i in range(M):
        Cov[i] = getCrossCov( Y[i], lag )  
    
    # hybrid thresholding 
    Edges = np.ones( (M, p,p) )
    
    # class-specific thresholding
    Edges[ abs(Cov) < lambda1 ] = 0

    # global thresholding   
    global_thresh =  np.sum( (Cov - lambda1 )**2 , 0 ) < lambda2
    for m in range(M):
        Edges[m][ global_thresh ] = 0
    
    # union of all connected parts across networks 
    Edges_Union = np.sum( Edges, axis=0 )
    ncc, cc = connected_components( Edges_Union )
    
    
    return ncc, cc


# fast network inference by identifying sub-graphs first
# the sub-graphs are identified using cross-covariance hybrid-thresholding
# the sub-graphs are then "merged" across experiments 
# so that we apply joint estimation to get connectivity estimates
# 

def net_inf_threshold( Y,  # required input of data, dictionary type, each is a Tm x P matrix
                      X= None , # integrated process/customized design matrix; 
                      lag = 1 ,# time lag when calculating the cross-covariance matrix 
                      lambda1 = 0.2, # threshold for class-specific thresholds 
                      lambda2 = 0 , # global thresholding, default 0 
                      msg = False # track progress of sub-graph learning
                     ):


    """
    Fast network inference by identifying sub-graphs first and applying the network inference procedure on each sub-graph. 
    The sub-graphs are identified using cross-covariance hybrid-thresholding (Tang et al 2015). 

    @param Y: dictionary of spiking/response matrix for each condition 
    @param X: dictionary of design matrix for each condition 
    @param lag: time lag used to calculate the cross-correlations
    @param lambda1: threshold for class-specific thresholds 
    @param lambda2: global thresholding, default is 0, meaning class-specific thresholding only
    @param msg: if True, print learning progress

    @return matrices of test statistic, p-values, lower and upper bound of the estimation 
    """


    M = len(Y)
    P = Y[0].shape[1]
    
    V   = np.zeros( (M, P,P) )
    PVAL = np.ones( (M, P,P) )
    LCI = np.zeros( (M, P,P) )
    UCI = np.zeros( (M, P,P) ) 

    # get connected components 
    ncc, cc = getConnectedComponentsHybrid(Y=Y, lag = lag, lambda1 =lambda1, lambda2 = lambda2)
    
    for k in range(ncc):
        if msg:
            print('start subgraph...', k+1, '...out of total', ncc)
        idx = np.where( cc ==k)
        idx = idx[0]

        Ysub = dict()
        for m in range(M):
            Ysub[m] = Y[m][:, idx] 
        if X is None:
            Xsub = None
        else:
            Xsub = dict()
            for m in range(M):
                Xsub[m] = X[m][ :, np.append(0, idx+1) ]

        v,pval, lci, uci = net_inf( Y= Ysub ,  X = Xsub )

        for i in range(len(idx)):
            for j in range(len(idx)):
                for m in range(M):
                    V[m][idx[i], idx[j]] =  v[m][i,j]
                    PVAL[m][idx[i], idx[j]] =  pval[m][i,j]
                    LCI[m][idx[i], idx[j]] =  lci[m][i,j]
                    UCI[m][idx[i], idx[j]] =  uci[m][i,j]


    return V, PVAL, LCI, UCI 

