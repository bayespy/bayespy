
# coding: utf-8

## Linear state-space model

# This example is also available as [an IPython notebook](lssm.ipynb) or [a Python script](lssm.py).
# 
# In linear state-space models a sequence of $M$-dimensional observations $\mathbf{Y}=(\mathbf{y}_1,\ldots,\mathbf{y}_N)$ is assumed to be generated from latent $D$-dimensional states $\mathbf{X}=(\mathbf{x}_1,\ldots,\mathbf{x}_N)$ which follow a first-order Markov process:
# 
# $$
# \mathbf{x}_{n} &= \mathbf{A}\mathbf{x}_{n-1} + \text{noise} \,,
# \\
# \mathbf{y}_{n} &= \mathbf{C}\mathbf{x}_{n} + \text{noise} \,,
# $$
# 
# where the noise is Gaussian, $\mathbf{A}$ is the $D\times D$ state dynamics matrix and $\mathbf{C}$ is the $M\times D$ loading matrix. Usually, the latent space dimensionality $D$ is assumed to be much smaller than the observation space dimensionality $M$ in order to model the dependencies of high-dimensional observations efficiently.
# 
# First, let us generate some toy data:

# In[1]:

import numpy as np

M = 30
N = 400

w = 0.3
a = np.array([[np.cos(w), -np.sin(w), 0, 0], 
              [np.sin(w), np.cos(w),  0, 0], 
              [0,         0,          1, 0],
              [0,         0,          0, 0]])
c = np.random.randn(M,4)
x = np.empty((N,4))
f = np.empty((M,N))
y = np.empty((M,N))
x[0] = 10*np.random.randn(4)
f[:,0] = np.dot(c,x[0])
y[:,0] = f[:,0] + 3*np.random.randn(M)
for n in range(N-1):
    x[n+1] = np.dot(a,x[n]) + np.random.randn(4)
    f[:,n+1] = np.dot(c,x[n+1])
    y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)
    


# 
# The linear state-space model can be constructed as follows:
# 

# In[2]:

from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
from bayespy.inference.vmp.nodes.gaussian import GaussianARD
from bayespy.inference.vmp.nodes.gamma import Gamma
from bayespy.inference.vmp.nodes.dot import SumMultiply

D = 10

# Dynamics matrix with ARD
alpha = Gamma(1e-5,
              1e-5,
              plates=(D,),
              name='alpha')
A = GaussianARD(0,
                alpha,
                shape=(D,),
                plates=(D,),
                name='A')

# Latent states with dynamics
X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                        1e-3*np.identity(D), # prec of x0
                        A,                   # dynamics
                        np.ones(D),          # innovation
                        n=N,                 # time instances
                        name='X',
                        initialize=False)
X.initialize_from_value(np.zeros((N,D))) # just some empty values, X is
                                         # updated first anyway

# Mixing matrix from latent space to observation space using ARD
gamma = Gamma(1e-5,
              1e-5,
              plates=(D,),
              name='gamma')
C = GaussianARD(0,
                gamma,
                shape=(D,),
                plates=(M,1),
                name='C')
# Initialize nodes (must use some randomness for C, and update X before C)
C.initialize_from_random()

# Observation noise
tau = Gamma(1e-5,
            1e-5,
            name='tau')

# Observations
F = SumMultiply('i,i',
                C, 
                X,
                name='F')
Y = GaussianARD(F,
                tau,
                name='Y')


# 
# An inference machine using variational Bayesian inference with variational
# message passing is then construced as
# 

# In[3]:

from bayespy.inference.vmp.vmp import VB
Q = VB(X, C, gamma, A, alpha, tau, Y)


# 
# Observe the data partially (80% is marked missing):
# 

# In[4]:

from bayespy.utils import random

# Add missing values randomly (keep only 20%)
mask = random.mask(M, N, p=0.2)
Y.observe(y, mask=mask)


# Then inference (100 iterations) can be run simply as

# In[5]:

Q.update(repeat=10)


### Speeding up with parameter expansion

# VB inference can converge extremely slowly if the variables are strongly coupled.  Because VMP updates one variable at a time, it may lead to slow zigzagging.  This can be solved by using parameter expansion which reduces the coupling. In state-space models, the states $\mathbf{x}_n$ and the loadings $\mathbf{C}$ are coupled through a dot product $\mathbf{Cx}_n$, which is unaltered if the latent space is rotated arbitrarily:
# 
# $$
# \mathbf{y}_n &= \mathbf{C}\mathbf{x}_n = \mathbf{C}\mathbf{R}^{-1}\mathbf{R}\mathbf{x}_n \,.
# $$
# 
# Thus, one intuitive transformation would be $\mathbf{C}\rightarrow\mathbf{C}\mathbf{R}^{-1}$ and $\mathbf{X}\rightarrow\mathbf{R}\mathbf{X}$.  In order to keep the dynamics of the latent states unaffected by the transformation, the state dynamics matrix $\mathbf{A}$ must be transformed accordingly:
# 
# $$
# \mathbf{R}\mathbf{x}_n &= \mathbf{R}\mathbf{A}\mathbf{R}^{-1} \mathbf{R}\mathbf{x}_{n-1} \,,
# $$
# 
# resulting in a transformation $\mathbf{A}\rightarrow\mathbf{R}\mathbf{A}\mathbf{R}^{-1}$.  For more details, refer to *Fast Variational Bayesian Linear State-Space Model (Luttinen, 2013).
# 
# In BayesPy, the transformations can be used as follows:

# In[6]:

# Import the parameter expansion module
from bayespy.inference.vmp import transformations

# Rotator of the state dynamics matrix
rotA = transformations.RotateGaussianARD(Q['A'], Q['alpha'])
# Rotator of the states (includes rotation of the state dynamics matrix)
rotX = transformations.RotateGaussianMarkovChain(Q['X'], rotA)
# Rotator of the loading matrix
rotC = transformations.RotateGaussianARD(Q['C'], Q['gamma'])
# Rotation optimizer
R = transformations.RotationOptimizer(rotX, rotC, D)


# Note that it is crucial to select the correct rotation class which corresponds
# to the particular model block exactly.  The rotation can be performed after each
# full VB update:

# In[7]:

for ind in range(10):
    Q.update()
    R.rotate()


# If you want to implement your own rotations or check the existing ones, you may
# use debugging utilities:

# In[8]:

for ind in range(10):
    Q.update()
    R.rotate(check_bound=True,
             check_gradient=True)

