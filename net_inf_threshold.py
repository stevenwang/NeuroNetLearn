# fast network learn using thresholding method

#  helper functions 
import numpy as np
from net_est_auto import getCrossCov
from net_inf import net_inf
from scipy.sparse.csgraph import connected_components

def getConnectedComponents( Y  , lag = 1 ,alpha =0.2):
    CC = [ ]
    NCC = [ ]
    for i in range(len(Y)):
        mat = getCrossCov( Y[i], lag )  
        mat[ abs(mat) < alpha ] = 0
        mat = 1.0*(mat!= 0)
        ncc, cc = connected_components( mat )
        NCC.append(ncc)
        CC.append(cc)
    
    return NCC, CC



def net_inf_threshold( Y,  # required input of data, dictionary type, each is a Tm x P matrix
                      X= None , # integrated process/customized design matrix; 
                      lag = 1 # time lag when calculating the cross-covariance matrix 
                     ):
    M = len(Y)
    P = Y[0].shape[1]
    
    V   = np.empty( (M, P,P) )
    PVAL = np.ones( (M, P,P) )
    LCI = np.empty( (M, P,P) )
    UCI = np.empty( (M, P,P) ) 

    # get connected components 
    NCC, CC = getConnectedComponents( Y , lag = lag)
    
    # learn at each subgraph
    for m in range( len(Y)):
        Ym = Y[m]
        ncc = NCC[m]
        cc = CC[m]
       
        for k in range(ncc):
            idx = np.where( cc ==k)
            idx = idx[0]

            Xm = None if not X else {0: X[m][ :, np.append(0, idx+1) ] }
 
            v,pval, lci, uci = net_inf( Y= {0:Ym[:, idx] } ,
                                        X = Xm )
            v=v[0]; pval = pval[0] ; lci = lci[0] ; uci = uci[0]

            for i in range(len(idx)):
                for j in range(len(idx)):
                    V[m][idx[i], idx[j]] =  v[i,j]
                    PVAL[m][idx[i], idx[j]] =  pval[i,j]
                    LCI[m][idx[i], idx[j]] =  lci[i,j]
                    UCI[m][idx[i], idx[j]] =  uci[i,j]
                    
        
    return V, PVAL, LCI, UCI 

