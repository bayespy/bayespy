
# coding: utf-8

## Quick start guide

# This short guide shows the key steps in using BayesPy for variational Bayesian inference by applying BayesPy to a simple problem. The key steps in using BayesPy are the following:
# 
# * Construct the model
# 
# * Observe some of the variables by providing the data in a proper format
# 
# * Run variational Bayesian inference
# 
# * Examine the resulting posterior approximation
# 
# To demonstrate BayesPy, we'll consider a very simple problem: we have a set of observations from a Gaussian distribution with unknown mean and variance, and we want to learn these parameters.  In this case, we do not use any real-world data but generate some artificial data. The dataset consists of ten samples from a Gaussian distribution with mean 5 and standard deviation 10. This dataset can be generated with NumPy as follows:

# In[1]:

import numpy as np
data = np.random.normal(5, 10, size=(10,))


# Now, given this data we would like to estimate the mean and the standard deviation as if we didn't know their values. The model can be defined as follows:
# 
# $$
# \begin{split}
# p(\mathbf{y}|\mu,\tau) &= \prod^{9}_{n=0} \mathcal{N}(y_n|\mu,\tau) \\
# p(\mu) &= \mathcal{N}(\mu|0,10^{-6}) \\
# p(\tau) &= \mathcal{G}(\tau|10^{-6},10^{-6})
# \end{split}
# $$
# 
# where $\mathcal{N}$ is the Gaussian distribution parameterized by its mean and precision (i.e., inverse variance), and $\mathcal{G}$ is the gamma distribution parameterized by its shape and rate parameters. Note that we have given quite uninformative priors for the variables $\mu$ and $\tau$.  This simple model can also be shown as a directed factor graph:

# This model can be constructed in BayesPy as follows:

# In[2]:

from bayespy.nodes import GaussianARD, Gamma
mu = GaussianARD(0, 1e-6)
tau = Gamma(1e-6, 1e-6)
y = GaussianARD(mu, tau, plates=(10,))


# In[3]:

y.observe(data)


# Next we want to estimate the posterior distribution.  In principle, we could use different inference engines (e.g., MCMC or EP) but currently only variational Bayesian (VB) engine is implemented.  The engine is initialized by giving all the nodes of the model:

# In[4]:

from bayespy.inference import VB
Q = VB(mu, tau, y)


# The inference algorithm can be run as long as wanted (max. 20 iterations in this case):

# In[5]:

Q.update(repeat=20)


# Now the algorithm converged after four iterations, before the requested 20 iterations. 
# 
# VB approximates the true posterior $p(\mu,\tau|\mathbf{y})$ with a distribution which factorizes with respect to the nodes: $q(\mu)q(\tau)$. The resulting approximate posterior distributions $q(\mu)$ and $q(\tau)$ can be examined, for instance, by plotting the marginal probability density functions:

# In[6]:

import bayespy.plot as bpplt
# The following two two lines are just for enabling matplotlib plotting in notebooks
get_ipython().magic('matplotlib inline')
bpplt.pyplot.plot([])
bpplt.pyplot.subplot(2, 1, 1)
bpplt.pdf(mu, np.linspace(-10, 20, num=100), color='k', name=r'\mu')
bpplt.pyplot.subplot(2, 1, 2)
bpplt.pdf(tau, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\tau');


# This example was a very simple introduction to using BayesPy.  The model can be much more complex and each phase contains more options to give the user more control over the inference.  The following sections give more details about the phases.
