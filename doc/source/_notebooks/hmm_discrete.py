
# coding: utf-8

## Discrete hidden Markov model

### Known parameters

# This example follows the one presented in [Wikipedia](http://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example). Each day, the state of the weather is either 'rainy' or 'sunny'. The weather follows a first-order discrete Markov process with the following initial state probability and state transition probabilities:

# In[43]:

from bayespy.nodes import CategoricalMarkovChain
# Initial state probabilities
a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
# State transition probabilities
A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
     [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
# The length of the process
N = 1000
# Markov chain
Z = CategoricalMarkovChain(a0, A, states=N)


# However, instead of observing this process directly, we observe whether Bob is 'walking', 'shopping' or 'cleaning'. The probability of each activity depends on the current weather as follows:

# In[44]:

from bayespy.nodes import Categorical, Mixture
# Emission probabilities
P = [[0.1, 0.4, 0.5],
     [0.6, 0.3, 0.1]]
# Observed process
Y = Mixture(Z, Categorical, P)


# In order to test our method, we'll generate artificial data using this model:

# In[45]:

# Draw realization of the weather process
weather = Z.random()
# Using this weather, draw realizations of the activities
activity = Mixture(weather, Categorical, P).random()


# Now, using this data, we set our variable $Y$ to be observed:

# In[46]:

Y.observe(activity)


# In order to run inference, we construct variational Bayesian inference engine:

# In[47]:

from bayespy.inference import VB
Q = VB(Y, Z)


# Note that we need to give all random variables to `VB`. In this case, the only random variables were `Y` and `Z`. Next we run the inference, that is, compute our posterior distribution:

# In[48]:

Q.update()


# In this case, because there is only one unobserved random variable, we recover the exact posterior distribution and there is no need to iterate more than one step.

### Unknown parameters

# Next, we consider the case when we do not know the parameters of the weather process (initial state probability and state transition probabilities). We give these parameters quite non-informative priors, but it is possible to provide more informative priors if such information is available. First, the weather process:

# In[49]:

from bayespy.nodes import Dirichlet
# Initial state probabilities
a0 = Dirichlet([0.1, 0.1])
# State transition probabilities
A = Dirichlet([[0.1, 0.1],
               [0.1, 0.1]])
# Markov chain
Z = CategoricalMarkovChain(a0, A, states=N)


# Second, the emission probabilities are also given quite non-informative priors:

# In[50]:

# Emission probabilities
P = Dirichlet([[0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1]])
# Observed process
Y = Mixture(Z, Categorical, P)


#  We use the same data as before:

# In[51]:

Y.observe(activity)


# Because `VB` takes all the unknown variables, we need to provide `A`, `a0` and `P` also:

# In[52]:

Q = VB(Y, Z, A, a0, P)


# If we ran the VB algorithm now, we would get a result where all both states would have identical emission probability distribution. This happens because of a non-random default initialization. `P` is initialized in such a way that both states have the same distribution, and `Z` is initialized in such a way that each state has equal probability. Thus, the VB algorithm won't separate them. In such cases, it is necessary to use a random initialization. In principle, it is possible to use random initialization for either variable and then update the other variable first. In the case of mixture distributions, it might be better to initialize the parameters (`P`) randomly and update the state assignments (`Z`) first.

# In[53]:

P.initialize_from_random()
Q.update(Z, A, a0, P, repeat=20)


# In order to update the variables in that order, one may explicitly give the nodes in that order to the `update` method. However, the default update order is the one used when constructing `Q`, which is the same in this case, thus we could have ignored listing the nodes to the `update` method.

# Plot the estimated state transition probabilities:

# In[55]:

# NOTE: These three lines are just to enable inline plotting in IPython Notebooks.
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot([])
# Plot the state transition matrix
import bayespy.plot.plotting as bpplt
bpplt.dirichlet_hinton(A)


# Plot the estimated emission probabilities:

# In[56]:

bpplt.dirichlet_hinton(P)


# It is interesting that these estimated parameters are very different from the true parameters. This happens because of un-identifiability: 
