# generate settings: circles, stars network

import scipy.linalg
import numpy as np
import math

def genCircleStarThreeNetworks(  m, star_coef, circle_coef, U_coef):
    
    """
    generate connectivity matrices of three simple networks of stars and circles 

    @param m: no. of blocks of starts or circles, minimum 4 blocks 
    @param star_coef: edge coefficient for star network
    @param circle_coef: edge coefficient for circle network
    @param U_coef: background intensity 
    
    @return background intensity matrices (3 x m*5), and connectivity matrices (3, m*5, m*5)
    """

    M=3 
    p=5 
    m = max( math.ceil(m/2)*2, 4)
    
    beta_circle = np.zeros( [p,p] )
    beta_circle[ 0, 1] = circle_coef
    beta_circle[ 1, 2] = circle_coef
    beta_circle[ 2, 3] = circle_coef
    beta_circle[3, 4] = circle_coef
    beta_circle[4, 0] = circle_coef
    
    beta_star = np.zeros( [p,p] )
    beta_star[ 0,2 ] = star_coef 
    beta_star[ 1,3 ] = star_coef
    beta_star[ 2, 4] = star_coef
    beta_star[3, 0] = star_coef
    beta_star[4, 1] = star_coef

    Beta = np.zeros( [M, p*m, p*m])
    Beta[0] = np.kron( np.eye(m) , beta_circle) 
    Beta[1] = np.kron( np.eye(2) , 
                       scipy.linalg.block_diag(   
                       beta_star, np.kron( np.eye( int(m/2) -1),beta_circle )  )      
                      ) 
    Beta[2] = np.kron( np.eye(m) , beta_star) 
    
    U = U_coef * np.ones( [ M, p*m] )
    

    return U , Beta




# random graph
def genRandomNetwork( p=50, edge=0.3, sparsity=0.02, pos_rate=1):  

    """
    generate a single network that is randomly connected 

    @param p: no. of nodes  
    @param edge: edge magnitute  
    @param sparsity: percentage of non-zero coefficients of the pxp connectivity matrix  
    @param pos_rate: percent of positive edges
    
    @return p by p connectivity matrix 
    """

    BETA = np.zeros( p*p )
    pos = np.random.choice(  p*p , int( p*p*sparsity*pos_rate), replace =False )
    BETA[pos] = np.abs(edge) 
    if pos_rate < 1:
        neg = np.random.choice( p*p, int( p*p*sparsity*(1-pos_rate) ), replace =False )
        BETA[neg] = -np.abs(edge) 
    
    BETA = BETA.reshape( (p,p) )
    idx = np.sum( np.abs(BETA), axis = 1) > min( 0.8, abs(edge)*3 )
    BETA[ idx, : ] = 0
    
    return BETA

