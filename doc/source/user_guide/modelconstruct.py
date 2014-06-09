
# coding: utf-8

## Constructing the model

### Creating nodes

# In[34]:

from bayespy.nodes import Gaussian
X = Gaussian([2, 5], [[1.0, 0.3], [0.3, 1.0]])


# or other nodes if they are unknown and given prior distributions:

# In[35]:

from bayespy.nodes import Gaussian, Wishart
mu = Gaussian([0, 0], [[1e-6, 0],[0, 1e-6]])
Lambda = Wishart(2, [[1, 0], [0, 1]])
X = Gaussian(mu, Lambda)


# However, there are two main restrictions for the parent nodes: non-constant parent nodes must be conjugate and the parent nodes must be independent in the posterior approximation.

#### Conjugacy of the parents

# In Bayesian framework in general, one can give quite arbitrary probability distributions for variables.  However, one often uses distributions that are easy to handle in practice.  Quite often this means that the parents are given conjugate priors.  This is also one of the limitations in BayesPy: only conjugate family prior distributions are accepted currently.  Thus, although in principle one could give, for instance, gamma prior for the mean parameter `mu`, only Gaussian-family distributions are accepted because of the conjugacy.  If the parent is not of a proper type, an error is raised. This conjugacy is checked automatically by BayesPy and `NoConverterError` is raised if a parent cannot be interpreted as being from a conjugate distribution.

#### Independence of the parents

# Another a bit rarely encountered limitation is that the parents must be independent (in the posterior factorization). Thus, a node cannot have the same stochastic node as several parents without intermediate stochastic nodes.  For instance, the following would lead to an error:

                
.. code-block:: python3

    from bayespy.nodes import Dot
    Y = Dot(X, X)

.. parsed-literal::

    ValueError: Parent nodes are not independent
                
# The error is raised because `X` is given as two parents for `Y`, and obviously `X` is not independent of `X` in the posterior approximation.  Even if `X` is not given several times directly but there are some intermediate deterministic nodes, an error is raised because the deterministic nodes depend on their parents and thus the parents of `Y` would not be independent.  However, it is valid that a node is a parent of another node via several paths if all the paths or all except one path has intermediate stochastic nodes.  This is valid because the intermediate stochastic nodes have independent posterior approximations.  Thus, for instance, the following construction does not raise errors:

# In[36]:

from bayespy.nodes import Dot
Z = Gaussian(X, [[1,0], [0,1]])
Y = Dot(X, Z)


# This works because there is now an intermediate stochastic node `Z` on the other path from `X` node to `Y` node.

### Effects of the nodes on inference

# When constructing the network with nodes, the stochastic nodes actually define three important aspects:
# 
#   1. the prior probability distribution for the variables,
#   
#   2. the factorization of the posterior approximation,
#   
#   3. the functional form of the posterior approximation for the variables.

#### Prior probability distribution

# $$
# p(\Omega) = \prod_{X \in \Omega} p(X|\pi_X),
# $$
# 
# where nodes correspond to the terms $p(X|\pi_X)$.

#### Posterior factorization

# Second, the nodes define the structure of the posterior approximation.  The variational Bayesian approximation factorizes with respect to nodes, that is, each node corresponds to an independent probability distribution in the posterior approximation.  In the previous example, `mu` and `tau` were separate nodes, thus the posterior approximation factorizes with respect to them: $q(\mu)q(\tau)$.  Thus, the posterior approximation can be written as:
# 
# $$
# p(\tilde{\Omega}|\hat{\Omega}) \approx \prod_{X \in \tilde{\Omega}} q(X),
# $$

#### Functional form of the posterior

### Using plate notation

#### Defining plates

# Stochastic nodes take the optional parameter `plates`, which can be used to define plates of the variable. A plate defines the number of repetitions of a set of variables. For instance, a set of random variables $\mathbf{y}_{mn}$ could be defined as
# 
# $$   
# \mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.
# $$
# 
# This can also be visualized as a graphical model:

# The variable has two plates: one for the index $m$ and one for the index $n$. In BayesPy, this random variable can be constructed as:

# In[37]:

y = Gaussian(mu, Lambda, plates=(10,30))


#### Sharing and broadcasting plates

# Instead of having a common mean and precision matrix for all $\mathbf{y}_{mn}$, it is also possible to share plates with parents. For instance, the mean could be different for each index $m$ and the precision for each index $n$:
# 
# $$
# \mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}_m, \mathbf{\Lambda}_n),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.
# $$
# 
# which has the following graphical representation:

# This can be constructed in BayesPy, for instance, as:

# In[38]:

from bayespy.nodes import Gaussian, Wishart
mu = Gaussian([0, 0], [[1e-6, 0],[0, 1e-6]], plates=(10,1))
Lambda = Wishart(2, [[1, 0], [0, 1]], plates=(1,30))
X = Gaussian(mu, Lambda)


# There are a few things to notice here. First, the plates are defined similarly as shapes in NumPy, that is, they use similar broadcasting rules. For instance, the plates `(10,1)` and `(1,30)` broadcast to `(10,30)`. In fact, one could use plates `(10,1)` and `(30,)` to get the broadcasted plates `(10,30)` because broadcasting compares the plates from right to left starting from the last axis. Second, `X` is not given `plates` keyword argument because the default plates are the plates broadcasted from the parents and that was what we wanted so it was not necessary to provide the keyword argument. If we wanted, for instance, plates `(20,10,30)` for `X`, then we would have needed to provide `plates=(20,10,30)`.
# 
# The validity of the plates between a child and its parents is checked as follows. The plates are compared plate-wise starting from the last axis and working the way forward.  A plate of the child is compatible with a plate of the parent if either of the following conditions is met:
# 
# 1. The two plates have equal size
# 2. The parent has size 1 (or no plate)
# 
# Table below shows an example of compatible plates for a child node and its two parent nodes:

#### Plates in deterministic nodes

# Note that plates can be defined explicitly only for stochastic nodes. For deterministic nodes, the plates are defined implicitly by the plate broadcasting rules from the parents. Deterministic nodes do not need more plates than this because there is no randomness. The deterministic node would just have the same value over the extra plates, but it is not necessary to do this explicitly because the child nodes of the deterministic node can utilize broadcasting anyway. Thus, there is no point in having extra plates in deterministic nodes, and for this reason, deterministic nodes do not use `plates` keyword argument.

#### Plates and shape

# It is useful to understand how the plates and the shape of a random variable are connected. The shape of an array which contains all the plates of a random variable is the concatenation of the plates and the shape of the variable. For instance, consider a 2-dimensional Gaussian variable with plates `(3,)`. If you want the value of the constant mean vector and constant precision matrix to vary between plates, they are given as `(3,2)`-shape and `(3,2,2)`-shape arrays, respectively:

# In[39]:

import numpy as np
mu = [ [0,0], [1,1], [2,2] ]
Lambda = [ [[1.0, 0.0],
            [0.0, 1.0]],
           [[1.0, 0.9],
            [0.9, 1.0]],
           [[1.0, -0.3],
            [-0.3, 1.0]] ]
X = Gaussian(mu, Lambda)
print("Shape of mu:", np.shape(mu))
print("Shape of Lambda:", np.shape(Lambda))
print("Plates of X:", X.plates)


# Thus, the leading axes of an array are the plate axes and the trailing axes are the random variable axes. In the example above, the mean vector has plates `(3,)` and shape `(2,)`, and the precision matrix has plates `(3,)` and shape `(2,2)`.

#### Factorization of plates

# It is important to undestand the independency structure the plates induce for the model. First, the repetitions defined by a plate are independent a priori given the parents. Second, the repetitions are independent in the posterior approximation, that is, the posterior approximation factorizes with respect to plates. Thus, the plates also have an effect on the independence structure of the posterior approximation, not only prior. If dependencies between a set of variables need to be handled, that set must be handled as a some kind of multi-dimensional variable.

#### Irregular plates

# The handling of plates is not always as simple as described above. There are cases in which the plates of the parents do not map directly to the plates of the child node. The user API should mention such irregularities.
# 
# For instance, the parents of a mixture distribution have a plate which contains the different parameters for each cluster, but the variable from the mixture distribution does not have that plate:

# In[40]:

from bayespy.nodes import Gaussian, Wishart, Categorical, Mixture
mu = Gaussian([[0], [0], [0]], [ [[1]], [[1]], [[1]] ])
Lambda = Wishart(1, [ [[1]], [[1]], [[1]]])
Z = Categorical([1/3, 1/3, 1/3], plates=(100,))
X = Mixture(Z, Gaussian, mu, Lambda)
print("Plates of mu:", mu.plates)
print("Plates of Lambda:", Lambda.plates)
print("Plates of Z:", Z.plates)
print("Plates of X:", X.plates)


# Also, sometimes the plates of the parents may be mapped to the variable axes. For instance, an automatic relevance determination (ARD) prior for a Gaussian variable is constructed by giving the diagonal elements of the precision matrix (or tensor). The Gaussian variable itself can be a scalar, a vector, a matrix or a tensor. A set of five $4 \times 3$ -dimensional Gaussian matrices with ARD prior is constructed as:

# In[41]:

from bayespy.nodes import GaussianARD, Gamma
tau = Gamma(1, 1, plates=(5,4,3))
X = GaussianARD(0, tau, shape=(4,3))
print("Plates of tau:", tau.plates)
print("Plates of X:", X.plates)


# Note how the last two plate axes of `tau` are mapped to the variable axes of `X` with shape `(4,3)` and the plates of `X` are obtained by taking the remaining leading plate axes of `tau`.
