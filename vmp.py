
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import profile


import imp
import gp
import nodes
import utils
imp.reload(gp)
imp.reload(nodes)
imp.reload(utils)
from gp import *
from nodes import *
from utils import *

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
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.fill_between(x,
                           Y[i]-L[i],
                           Y[i]+U[i],
                           facecolor=(0.6,0.6,0.6,1),
                           edgecolor=(0,0,0,0),
                           linewidth=0)
        plt.plot(x, Y[i], color=(0,0,0,1))
        plt.ylabel(str(i))



# MULTIVARIATE GP!!

def test_gp():

    # Generate data
    x = np.random.uniform(low=0, high=10, size=(100,))
    f = np.sin(x*2*np.pi/5)
    f = f + np.random.normal(0, 0.2, np.shape(f))
    plt.clf()
    plt.plot(x,f,'r+')
    #plt.plot(x,y,'r+')

    # Construct model
    ls = NodeConstantScalar(1.1, name='lengthscale')
    amp = NodeConstantScalar(1.0, name='amplitude')
    noise = NodeConstantScalar(1.3, name='noise')
    K = NodeCovarianceFunctionSE(amp, ls)
    K_noise = NodeCovarianceFunctionDelta(noise)
    K_sum = NodeCovarianceFunctionSum(K, K_noise)
    M = NodeConstantGaussianProcess(lambda x: (x/10-2)*(x/10+1))
    F = NodeGaussianProcess(M, K_sum)

    # Inference
    F.observe(x, f)
    vb_optimize_nodes(ls, amp, noise)
    F.update()
    u = F.get_parameters()

    print('parameters')
    print(ls.name)
    print(ls.u[0])
    print(amp.name)
    print(amp.u[0])
    print(noise.name)
    print(noise.u[0])

    #print(F.lower_bound_contribution())

    # Posterior predictions
    xh = np.arange(-5, 20, 0.1)
    (fh, varfh) = u(xh, covariance=1)

    #print(fh)
    #print(varfh)

    errfh = np.sqrt(varfh)
    ## print(np.shape(xh))
    ## print(np.shape(fh))
    #print(varfh[-1])
    ## print(np.shape(errfh))
    ## print(errfh)
    m_errorplot(xh, fh, errfh, errfh)
    
    return
    
    # Construct a GP
    k = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    f = NodeGP(0, k)
    f.observe(x, y)
    f.update()
    (mp, kp) = f.get_parameters()


def test_pca():

    # Dimensionalities
    dataset = 1
    if dataset == 1:
        M = 10
        N = 100
        D_y = 3
        D = 3+2
        # Generate data
        w = np.random.normal(0, 1, size=(M,1,D_y))
        x = np.random.normal(0, 1, size=(1,N,D_y))
        f = sum_product(w, x, axes_to_sum=[-1])#np.einsum('...i,...i', w, x)
        y = f + np.random.normal(0, 0.5, size=(M,N))
    elif dataset == 2:
        # Data from matlab comparison
        f = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_f.txt')
        y = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_y.txt')
        D = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_d.txt')
        (M,N) = np.shape(y)

    # Construct the PCA model with ARD

    alpha = NodeGamma(1e-10, 1e-10, plates=(D,), name='alpha')
    alpha.update()
    diag_alpha = NodeWishartFromGamma(alpha)
    #Lambda = NodeWishart(D, D * np.identity(D), plates=(), name='Lambda')
    #Lambda.update()
    
    X = NodeGaussian(np.zeros(D), np.identity(D), name="X", plates=(1,N))

    X.update()
    X.u[0] = X.random()

    W = NodeGaussian(np.zeros(D), diag_alpha, name="W", plates=(M,1))
    #W = NodeGaussian(np.zeros(D), Lambda, name="W", plates=(M,1))
    #W = NodeGaussian(np.zeros(D), np.identity(D), name="W", plates=(M,1))
    W.update()
    W.u[0] = W.random()

    WX = NodeDot(W,X,S,R)

    tau = NodeGamma(1e-5, 1e-5, name="tau", plates=(M,N))
    tau.update()

    Y = NodeNormal(WX, tau, name="Y", plates=(M,N))
    Y.update()

    # Initialize (from prior)

    # Y.update()
    # mask = True
    # mask = np.ones((M,N), dtype=np.bool)
    mask = np.random.rand(M,N) < 0.4
    mask[:,20:40] = False
    Y.observe(y, mask)

    # Inference
    L_last = -np.inf
    for i in range(100):
        t = time.clock()
        X.update()
        W.update()
        tau.update()
        #Lambda.update()
        alpha.update()

        L_X = X.lower_bound_contribution()
        L_W = W.lower_bound_contribution()
        L_tau = tau.lower_bound_contribution()
        L_Y = Y.lower_bound_contribution()
        L_alpha = alpha.lower_bound_contribution()
        #print("X: %f, W: %f, tau: %f, Y: %f" % (L_X, L_W, L_tau, L_Y))
        L = L_X + L_W + L_tau + L_Y
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            #raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-12:
            print("Converged.")
            #break
        L_last = L

    #return


    #print(shape(yh))
    plt.figure(1)
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    m_errorplot(np.arange(N), fh, err_fh, err_fh)
    m_plot(np.arange(N), f, 'g')
    m_plot(np.arange(N), y, 'r+')

    #plt.figure(2)
    #plt.clf()

    #alpha.show()
    print(alpha.u[0])
    
        
    tau.show()



def test_normal():

    M = 10
    N = 5

    # mu
    mu = NodeNormal(0.0, 10**-5, name="mu", plates=())
    print("Prior for mu:")
    mu.update()
    mu.show()

    # tau
    tau = NodeGamma(10**-5, 10**-5, plates=(N,), name="tau")
    print("Prior for tau:")
    tau.update()
    tau.show()

    # x
    x = NodeNormal(mu, tau, plates=(M,N), name="x")
    print("Prior for x:")
    x.update()
    x.show()

    # y (generate data)
    y = NodeNormal(x, 1, plates=(M,N), name="y")
    y.observe(random.normal(loc=10, scale=10, size=(M,N)))

    # Inference
    for i in range(50):
        x.update()
        mu.update()
        tau.update()

    print("Posterior for mu:")
    mu.show()
    print("Posterior for tau:")
    tau.show()
    print("Posterior for x:")
    x.show()
    
    return
    
def test_multivariate():    

    D = 3
    N = 100
    M = 200

    # mu
    mu = NodeGaussian(np.zeros(D), 10**(-10)*np.identity(D), plates=(M,1), name='mu')
    print("Prior for mu:")
    mu.update()
    mu.show()

    # Lambda
    Lambda = NodeWishart(D, (10**-10) * np.identity(D), plates=(1,N), name='Lambda')
    print("Prior for Lambda:")
    Lambda.update()
    Lambda.show()

    #Y = NodeGaussian(mu, 10**(-2)*identity(D), plates=(M,N), name='Y')
    Y = NodeGaussian(mu, Lambda, plates=(M,N), name='Y')
    Y.observe(random.normal(loc=10, scale=10, size=(M,N,D)))

    ## # y (generate data)
    ## for i in range(100):
    ##     y = NodeGaussian(mu, Lambda)
    ##     v = random.normal(0,1, D)
    ##     y.fix(v)

    # Inference
    try:
        for i in range(50):
            mu.update()
            Lambda.update()

        print("Posterior for mu:")
        mu.show()
        print("Posterior for Lambda:")
        Lambda.show()
    except Exception:
        pass
    

if __name__ == '__main__':

    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()

    test_gp()
    #test_pca()
    #profile.run('test_pca()', 'profile.tmp')
    #test_normal()
    #test_multivariate()

