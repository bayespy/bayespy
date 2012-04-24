
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
#from matplotlib.pyplot import *

def matrixplot(A, colorbar=False):
    if sp.issparse(A):
        A = A.toarray()
    plt.imshow(A, interpolation='nearest')
    if colorbar:
        plt.colorbar()


def contourplot(x1, x2, y, colorbar=False, filled=True):
    """ Plots 2D contour plot. x1 and x2 are 1D vectors, y contains
    the function values. y.size must be x1.size*x2.size. """
    
    y = np.reshape(y, (len(x2),len(x1)))
    if filled:
        plt.contourf(x1, x2, y)
    else:
        plt.contour(x1, x2, y)
    if colorbar:
        plt.colorbar()
        

def errorplot(y, error=None, x=None, lower=None, upper=None):

    # Default inputs
    if x is None:
        x = np.arange(np.size(y))

    # Parse errors (lower=lower/error/upper, upper=upper/error/lower)
    if lower is None:
        if error is not None:
            lower = error
        elif upper is not None:
            lower = upper
    if upper is None:
        if error is not None:
            upper = error
        elif lower is not None:
            upper = lower

    # Plot errors
    if (lower is not None) and (upper is not None):
        print(np.max(lower))
        print(np.max(upper))
        l = y - lower
        u = y + upper
        plt.fill_between(x,
                         u,
                         l,
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         linewidth=0,
                         interpolate=True)
    # Plot function
    plt.plot(x, y, color=(0,0,0,1))

#def multiplot(plot_function, *args, **kwargs):
    

def m_plot(x, Y, style):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.plot(x, Y[i], style)

## def multi_errorplot(Y, error=None, x=None, lower=None, upper=None):

##     for m in range(M):
##         for n in range(N):
##             plt.subplot(M,N,m*N+n)
##             errorplot(Y[m][n],
##                       error=error[m][n],
##                       x=x[m][n],
##                       lower=lower[m][n],
##                       upper=upper[m][n])

def m_errorplot(x, Y, L, U):
    Y = np.atleast_2d(Y)
    L = np.atleast_2d(L)
    U = np.atleast_2d(U)
    M = Y.shape[-2]
    ## print(np.shape(Y))
    ## print(np.shape(L))
    ## print(np.shape(U))
    ## print(np.shape(M))
    for i in range(M):
        plt.subplot(M,1,i+1)
        lower = Y[i] - L[i]
        upper = Y[i] + U[i]
        #print(upper-lower)
        #if np.any(lower>=upper):
            #print('WTF?!')
        plt.fill_between(x,
                         upper,
                         lower,
                         #where=(upper>=lower),
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         #edgecolor=(0.6,0.6,0.6,1),
                         linewidth=0,
                         interpolate=True)
        plt.plot(x, Y[i], color=(0,0,0,1))
        plt.ylabel(str(i))

