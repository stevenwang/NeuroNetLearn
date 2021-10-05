# utility functions 



# plot edge matrices into graphs 
import numpy as np
from igraph import *
import matplotlib.pyplot as plt 

def plot_graph( Edges, # edge or coef matrix for each experiment, multi-dim array (M, p , p)
               rc = None ,  # no. figure at each row and column,
               layout = None 
              ):
    """
    plot edge matrices into graphs 

    @param Edges: edge or coef matrix for each experiment, multi-dim array (M, p , p)
    @param rc: optional number of figures at each row and column
    @param layout: optional layout type extracted from existing igraph Graph

    @return layout object associated with the plotted Graph; if layout is provided, then nothing is returned
    """
    Edges = (Edges !=0)*1.0

    if len(Edges.shape) == 2:
        Edges = np.array( [Edges])

    M = Edges.shape[0]
        
    if rc is None:
        c = min(3, M)
        r = max( 1, int(M/ c) )
        rc= ( r, c )
    
    fig, axs = plt.subplots( rc[0], rc[1] ,figsize=(6*M, 6))  
    
    if M ==1:
        axs= np.array( [axs])

    layout_return = False 

    for m in range(M):
        axs[m].axis('off')

        g = Graph.Adjacency(Edges[m])
        if m ==0 and layout is None:
            layout = g.layout(layout='auto')
            layout_return = True 
        plot(g,  layout = layout ,              
                 vertex_color='pink', 
                 vertex_label = np.arange(Edges[m].shape[0]), 
                 vertex_size = 8, 
                 target= axs[m] )

        axs[m].set_title('Sig. Edge. of Network:'+ str(m+1) ,fontsize=20)

    #print(layout_return)
    if layout_return:
    	return layout 

