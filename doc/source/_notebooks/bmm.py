
# coding: utf-8

## Bernoulli mixture model

# blaa blaa blaa

# In[1]:

import numpy as np
D = 10
p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
p1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
p = np.array([p0, p1, p2])
from bayespy.utils import random
N = 100
z = random.categorical([1/3, 1/3, 1/3], size=N)
x = random.bernoulli(p[z])


# In[2]:

from bayespy.nodes import Categorical, Dirichlet
K = 5
R = Dirichlet(K*[1e-3],
              name='R')
Z = Categorical(R,
                plates=(N,1),
                name='Z')

from bayespy.nodes import Mixture, Bernoulli, Beta
P = Beta([1e-1, 1e-1],
         plates=(D,K),
         name='P')
X = Mixture(Z, Bernoulli, P)

X.observe(x)

from bayespy.inference import VB
Q = VB(Z, R, X, P)
P.initialize_from_random()

Q.update(repeat=10)


# In[2]:

from bayespy.plot.plotting as bpplt
bpplt.

