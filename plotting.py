
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

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
    if not x:
        x = np.arange(np.size(y))

    # Parse errors (lower=lower/error/upper, upper=upper/error/lower)
    if lower == None:
        if error != None:
            lower = error
        elif upper != None:
            lower = upper
    if upper == None:
        if error != None:
            upper = error
        elif lower != None:
            upper = lower

    # Plot errors
    if (lower != None) and (upper != None):
        l = y - lower
        u = y + upper
        plt.fill_between(x,
                         upper,
                         lower,
                         facecolor=(0.6,0.6,0.6,1),
                         edgecolor=(0,0,0,0),
                         linewidth=0,
                         interpolate=True)
    # Plot function
    plt.plot(x, y, color=(0,0,0,1))

def m_plot(x, Y, style):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.plot(x, Y[i], style)

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

