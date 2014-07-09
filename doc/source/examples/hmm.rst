..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public
   License. See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 3 as
   published by the Free Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.

.. testsetup::

   import numpy
   numpy.random.seed(1)

Discrete hidden Markov model
============================


Known parameters
----------------


Model
+++++

This example follows the one presented in `Wikipedia
<http://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example>`__.  Each
day, the state of the weather is either 'rainy' or 'sunny'. The weather follows
a first-order discrete Markov process with the following initial state
probability and state transition probabilities:

>>> from bayespy.nodes import CategoricalMarkovChain

Initial state probabilities

>>> a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
    
State transition probabilities

>>> A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
...      [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
    
The length of the process

>>> N = 100
    
Markov chain

>>> Z = CategoricalMarkovChain(a0, A, states=N)

However, instead of observing this process directly, we observe whether Bob is
'walking', 'shopping' or 'cleaning'. The probability of each activity depends on
the current weather as follows:

>>> from bayespy.nodes import Categorical, Mixture

Emission probabilities

>>> P = [[0.1, 0.4, 0.5],
...      [0.6, 0.3, 0.1]]

Observed process

>>> Y = Mixture(Z, Categorical, P)

Data
++++

In order to test our method, we'll generate artificial data using this model:

Draw realization of the weather process

>>> weather = Z.random()

Using this weather, draw realizations of the activities

>>> activity = Mixture(weather, Categorical, P).random()

Inference
+++++++++

Now, using this data, we set our variable :math:`Y` to be observed:

>>> Y.observe(activity)

In order to run inference, we construct variational Bayesian inference engine:

>>> from bayespy.inference import VB
>>> Q = VB(Y, Z)

Note that we need to give all random variables to ``VB``. In this case, the only
random variables were ``Y`` and ``Z``. Next we run the inference, that is,
compute our posterior distribution:

>>> Q.update()
Iteration 1: loglike=-1.095883e+02 (... seconds)

In this case, because there is only one unobserved random variable, we
recover the exact posterior distribution and there is no need to iterate
more than one step.

Results
+++++++

One way to plot the categorical timeseries is to use the Hinton diagram:

>>> import bayespy.plot as bpplt
>>> bpplt.hinton(Z, square=False)

.. plot::

   import numpy
   numpy.random.seed(1)
   from bayespy.nodes import CategoricalMarkovChain
   a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
   A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
        [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
   N = 100
   Z = CategoricalMarkovChain(a0, A, states=N)
   from bayespy.nodes import Categorical, Mixture
   P = [[0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]]
   Y = Mixture(Z, Categorical, P)
   weather = Z.random()
   activity = Mixture(weather, Categorical, P).random()
   Y.observe(activity)
   from bayespy.inference import VB
   Q = VB(Y, Z)
   Q.update()
   import bayespy.plot as bpplt
   bpplt.hinton(Z, square=False)
   bpplt.pyplot.show()

Non-square blocks are squeezed to fit appropriately in the plot.  Time axis is
vertical and the two states are side by side.  The wider the white bar, the more
probable the state at that time is.



Unknown parameters
------------------

In this example, we consider unknown parameters for the Markov process and
different emission distribution.

Model
+++++

Now, we do not know the parameters of the weather process (initial state
probability and state transition probabilities). We give these parameters quite
non-informative priors, but it is possible to provide more informative priors if
such information is available. First, the weather process:

>>> from bayespy.nodes import Dirichlet

Initial state probabilities

>>> a0 = Dirichlet([0.1, 0.1])

State transition probabilities

>>> A = Dirichlet([[0.1, 0.1],
...                [0.1, 0.1]])

Markov chain

>>> Z = CategoricalMarkovChain(a0, A, states=N)

Second, the emission probabilities are also given quite non-informative priors:

Emission probabilities

>>> P = Dirichlet([[0.1, 0.1, 0.1],
...                [0.1, 0.1, 0.1]])

Observed process

>>> Y = Mixture(Z, Categorical, P)

Inference
+++++++++

We use the same data as before:

>>> Y.observe(activity)

Because ``VB`` takes all the unknown variables, we need to provide ``A``, ``a0``
and ``P`` also:

>>> Q = VB(Y, Z, A, a0, P)

If we ran the VB algorithm now, we would get a result where all both states
would have identical emission probability distribution. This happens because of
a non-random default initialization. ``P`` is initialized in such a way that
both states have the same distribution, and ``Z`` is initialized in such a way
that each state has equal probability. Thus, the VB algorithm won't separate
them. In such cases, it is necessary to use a random initialization. In
principle, it is possible to use random initialization for either variable and
then update the other variable first. In the case of mixture distributions, it
might be better to initialize the parameters (``P``) randomly and update the
state assignments (``Z``) first.

>>> P.initialize_from_random()
>>> Q.update(Z, A, a0, P, repeat=1000)
Iteration 1: loglike=-1.293357e+02 (... seconds)
...
Iteration 38: loglike=-1.229328e+02 (... seconds)
Converged at iteration 38.

In order to update the variables in that order, one may explicitly give the
nodes in that order to the ``update`` method. However, the default update order
is the one used when constructing ``Q``, which is the same in this case, thus we
could have ignored listing the nodes to the ``update`` method.

Results
+++++++

Let us plot the estimated parameters. First, the state transition matrix:

>>> bpplt.hinton(A)

.. plot::

   import numpy
   numpy.random.seed(1)
   from bayespy.nodes import CategoricalMarkovChain
   a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
   A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
        [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
   N = 100
   Z = CategoricalMarkovChain(a0, A, states=N)
   from bayespy.nodes import Categorical, Mixture
   P = [[0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]]
   Y = Mixture(Z, Categorical, P)
   weather = Z.random()
   activity = Mixture(weather, Categorical, P).random()
   Y.observe(activity)
   from bayespy.inference import VB
   Q = VB(Y, Z)
   Q.update()
   import bayespy.plot as bpplt
   from bayespy.nodes import Dirichlet
   a0 = Dirichlet([0.1, 0.1])
   A = Dirichlet([[0.1, 0.1],
                  [0.1, 0.1]])
   Z = CategoricalMarkovChain(a0, A, states=N)
   P = Dirichlet([[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1]])
   Y = Mixture(Z, Categorical, P)
   Y.observe(activity)
   Q = VB(Y, Z, A, a0, P)
   P.initialize_from_random()
   Q.update(Z, A, a0, P, repeat=1000)
   bpplt.hinton(A)
   bpplt.pyplot.show()


Second, the emission probabilities:

>>> bpplt.hinton(P)

.. plot::

   import numpy
   numpy.random.seed(1)
   from bayespy.nodes import CategoricalMarkovChain
   a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
   A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
        [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
   N = 100
   Z = CategoricalMarkovChain(a0, A, states=N)
   from bayespy.nodes import Categorical, Mixture
   P = [[0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]]
   Y = Mixture(Z, Categorical, P)
   weather = Z.random()
   activity = Mixture(weather, Categorical, P).random()
   Y.observe(activity)
   from bayespy.inference import VB
   Q = VB(Y, Z)
   Q.update()
   import bayespy.plot as bpplt
   from bayespy.nodes import Dirichlet
   a0 = Dirichlet([0.1, 0.1])
   A = Dirichlet([[0.1, 0.1],
                  [0.1, 0.1]])
   Z = CategoricalMarkovChain(a0, A, states=N)
   P = Dirichlet([[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1]])
   Y = Mixture(Z, Categorical, P)
   Y.observe(activity)
   Q = VB(Y, Z, A, a0, P)
   P.initialize_from_random()
   Q.update(Z, A, a0, P, repeat=1000)
   bpplt.hinton(P)
   bpplt.pyplot.show()

Note that these estimated parameters are very different from the true
parameters. This happens because of un-identifiability (different parameters
lead to similar marginal distributions over the observed process) and the data
does not have enough evidence for the true parameters.  Note also that the
states in this model do not anymore correspond to weather.  We can plot the
states similarly as in the fixed parameter case:


>>> bpplt.hinton(Z, square=False)

.. plot::

   import numpy
   numpy.random.seed(1)
   from bayespy.nodes import CategoricalMarkovChain
   a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
   A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
        [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
   N = 100
   Z = CategoricalMarkovChain(a0, A, states=N)
   from bayespy.nodes import Categorical, Mixture
   P = [[0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]]
   Y = Mixture(Z, Categorical, P)
   weather = Z.random()
   activity = Mixture(weather, Categorical, P).random()
   Y.observe(activity)
   from bayespy.inference import VB
   Q = VB(Y, Z)
   Q.update()
   import bayespy.plot as bpplt
   from bayespy.nodes import Dirichlet
   a0 = Dirichlet([0.1, 0.1])
   A = Dirichlet([[0.1, 0.1],
                  [0.1, 0.1]])
   Z = CategoricalMarkovChain(a0, A, states=N)
   P = Dirichlet([[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1]])
   Y = Mixture(Z, Categorical, P)
   Y.observe(activity)
   Q = VB(Y, Z, A, a0, P)
   P.initialize_from_random()
   Q.update(Z, A, a0, P, repeat=1000)
   bpplt.hinton(Z, square=False)
   bpplt.pyplot.show()

This example could be modified by using some other emission distribution.  For instance, Gaussian distribution would lead to a Gaussian mixture model with dynamics.
