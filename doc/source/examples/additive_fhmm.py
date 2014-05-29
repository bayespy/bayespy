
# coding: utf-8

## Additive factorial hidden Markov model

# In[5]:

import numpy as np
from bayespy.nodes import (Dirichlet, 
                            CategoricalMarkovChain,
                            GaussianARD,
                            Gate,
                            SumMultiply,
                            Gamma)

D = 4   # dimensionality of the data vectors (and mu vectors)
N = 5   # number of chains
K = 3   # number of states in each chain
T = 100 # length of each chain

# Markov chain parameters.
# Use known values
p0 = np.ones(K) / K
P = np.ones((K,K)) / K
# Or set Dirichlet prior.
p0 = Dirichlet(np.ones(K), plates=(N,))
P = Dirichlet(np.ones(K), plates=(N,1,K))

# N Markov chains with K possible states, and length T
X = CategoricalMarkovChain(p0, P, states=T, plates=(N,))

# For each of the N chains, have K different D-dimensional mu's
# Unknown mu's
mu = GaussianARD(0, 1e-3, plates=(D,1,1,K), shape=(N,))

# Gate/select mu's
print(mu.plates, mu.dims[0], X.plates, X.dims[1])
Z = Gate(X, mu)

# Sum the mu's of different chains
F = SumMultiply('i->', Z)
print(mu.plates, mu.dims[0], X.plates, X.dims[0], Z.plates, Z.dims[0], F.plates, F.dims[0])

# Known observation noise inverse covariance
tau = np.ones(D)
# or unknown observation noise inverse covariance
tau = Gamma(1e-3, 1e-3, plates=(D,))

# Observed process
Y = GaussianARD(F, tau)

# Data
data = np.random.randn(T, D)
Y.observe(data)

from bayespy.inference import VB
Q = VB(Y, X, p0, P, mu, tau)

Q.update(repeat=10)


# In[7]:




# In[ ]:



