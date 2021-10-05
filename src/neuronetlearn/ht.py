#################################
# hierarchical testing procedure 
#
# input: 
# X: p x M matrix, test statistics matrix for each coefficients (total of p), 
# e.g, for high-dimensional point process, use decorrelated score statistics 
# Note that, the colums of X needs to be ordered in a way following the similarity hierarchy
# before use, where the 1st col correponds to the leftest node in the binary tree 
# and the last col corresponds to the bottom right corner coefficient
# Output:
# R: rejection matrix, each element of R referes to 1/0 reject or not for testing each coefficient from 0
# P: the corresponding p-values for testing each coefficient from 0

# Algorithm outline:
# the root of the binary tree is called level 1, 
# any level with 2 nodes has its level from 2 to M
# the binary tree are constructed following a hierarchy of similarity
# in each level, we can decide the condition rejection based on left node 
# we use the right node to decide move downward or not 


import numpy as np
from scipy.stats import chi2

def prepareTestStatMat( X, h_order= None ):

    """
    prepare test statistics matrix used to be compatible with the HT function

    @param X: p x M matrix, test statistics matrix for each coefficients (total of p), 
    e.g, for high-dimensional point process, use decorrelated score statistics. 
    Note that, the colums of X needs to be ordered in a way following the similarity hierarchy
    before use, where the 1st col correponds to the leftest node in the binary tree 
    and the last col corresponds to the bottom right corner coefficient 
    @param h_order: if provided, columns of X is reorder to represent the similarity hierarchy 
   
    @return test statistics matrix prepared for the hierarchical testing function
    """


    if len(X.shape) == 1:
        msg = 'reshaped from 1D to 2D'
        p = X.shape[0] 
        X = X.reshape( (1, p) )

    if len(X.shape)==2:
        msg = 'already in 2D, no reshape performed' 

    if len( X.shape) == 3:
        msg = 'reshaped from 3D to 2D'
        M, p, _ = X.shape
        X = X.reshape( (M, -1 ) )
        X = X.T 

    if len(X.shape) >3:
        msg = 'dimension can be larger than 3' 

    # reorder
    if h_order:
        msg = msg + '; reordered columns according to h_order'
        X = X[: , h_order ]
    else:
        msg = msg + '; use default order'

    print( msg )

    return X



def HT( X,  alpha = 0.05 ):

    """
    Hierarchical testing procedure. 
    When the columns of the input matrix is ordered following the similarity hierarchy,
    the procedure reduces the number of tests involved thus is more powerful 
    At the same time, it controls the family-wise error rate (FWER).

    @param X: p x M matrix, test statistics matrix for each coefficients (total of p), 
    e.g, for high-dimensional point process, use decorrelated score statistics. 
    Note that, the colums of X needs to be ordered in a way following the similarity hierarchy
    before use, where the 1st col correponds to the leftest node in the binary tree 
    and the last col corresponds to the bottom right corner coefficient 
    @param alpha: family-wise error rate level

    @return rejection matrix and p-value matrix calculated on available hierarchy path.
    """
    
    p, M = X.shape
    
    R = np.zeros( (p, M) ) # each element of R referes to 1/0 reject or not status for each coefficient
    P = np.zeros( (p, M) ) # each element of P referes to the p-value of testing each coefficient

    pv_right = 1- chi2.cdf(  np.sum( X**2 , axis= 1) ,df=M)
    rej_last =  1.0*( pv_right < alpha/ p )
    
    for l in range(1,M):
        if M < 2:
            break 
        X_l  =  X[: , (l-1):] 

        pv_left  =1 - chi2.cdf( X_l[:,0]**2 ,df=1)  
        pv_right =1 - chi2.cdf(  np.sum( X_l[:,1:]**2 , axis= 1) ,df=M-l) 

        C_l = (M-l) # under oracle hierarchical clustering 
        cval = alpha/p * C_l/ M # critical value 

        P[:,l-1] = pv_left
        R[:,l-1] = ( pv_left < cval ) * rej_last # the more left columns in R refer to left leaf hypothesis
        rej_last =( pv_right < cval ) * rej_last

        P[:,M-1] = pv_right 
        R[:,M-1] = rej_last


    return R, P










