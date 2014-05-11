
# coding: utf-8

## Discrete hidden Markov model

# In[1]:

from bayespy.nodes import CategoricalMarkovChain, Categorical, Mixture
N = 100
# Initial state probabilities
a0 = [0.6, 0.4]
# State transition probabilities
A = [[0.7, 0.3],[0.4,0.6]]
# Markov chain
Z = CategoricalMarkovChain(a0, A, states=N)
# Emission probabilities
P = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]
# Observed process
Y = Mixture(Z, Categorical, P)
# Generate dummy data
import numpy as np
data = np.random.randint(3, size=N)
# Observe data
Y.observe(data)

from bayespy.inference import VB
Q = VB(Y, Z)

Q.update()

