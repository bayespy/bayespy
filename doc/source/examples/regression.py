
# coding: utf-8

## Regression

### Linear regression

# In[1]:

import numpy as np
k = 2
c = 5
s = 2

x = np.arange(10)
y = k*x + c + s*np.random.randn(10)

from bayespy.nodes import GaussianARD
B = GaussianARD(0, 1e-6, shape=(2,))

X = np.vstack([x, np.ones(len(x))]).T

from bayespy.nodes import SumMultiply
F = SumMultiply('i,i', B, X)

from bayespy.nodes import Gamma
tau = Gamma(1e-3, 1e-3)

Y = GaussianARD(F, tau)
Y.observe(y)

from bayespy.inference import VB
Q = VB(Y, B, tau)

Q.update(repeat=100)

import bayespy.plot as bpplt
# These two lines are needed to enable inline plotting IPython Notebooks
get_ipython().magic('matplotlib inline')
bpplt.pyplot.plot([])

xh = np.linspace(-5, 15, 100)
Xh = np.vstack([xh, np.ones(len(xh))]).T
Fh = SumMultiply('i,i', B, Xh)
bpplt.plot(Fh, x=xh, scale=2)
bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
bpplt.plot(k*xh+c, x=xh, color='r')


# In[2]:

bpplt.pdf(tau, np.linspace(0,1,100), color='k')
bpplt.pyplot.axvline(s**(-2), color='r')


# In[3]:

bpplt.contour(B, np.linspace(1,3,100), np.linspace(1,9,100), n=10, colors='k')
bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2)


### Improving accuracy

# In[4]:

from bayespy.nodes import GaussianGammaISO
B_tau = GaussianGammaISO(np.zeros(2), 1e-6*np.identity(2), 1e-3, 1e-3)

F_tau = SumMultiply('i,i', B_tau, X)

Y = GaussianARD(F_tau, 1)
Y.observe(y)

from bayespy.inference import VB
Q = VB(Y, B_tau)

Q.update(repeat=10)


# In[5]:

#import bayespy.plot as bpplt
# These two lines are needed to enable inline plotting IPython Notebooks
#%matplotlib inline
#bpplt.plt.plot([])

bpplt.plotmatrix(B_tau)


# In[6]:

xh = np.linspace(-5, 15, 100)
Xh = np.vstack([xh, np.ones(len(xh))]).T
Fh = SumMultiply('i,i', B, Xh)
bpplt.timeseries(Fh, x=xh, scale=2)
bpplt.plt.plot(x, y, 'rx')
bpplt.plt.plot(xh, k*xh+c, 'r')


### Multivariate regression

### Non-linear regression

# In[6]:




# In[6]:



