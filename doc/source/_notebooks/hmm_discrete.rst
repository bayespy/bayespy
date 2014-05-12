
Discrete hidden Markov model
============================

Known parameters
----------------

This example follows the one presented in
`Wikipedia <http://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example>`_.
Each day, the state of the weather is either 'rainy' or 'sunny'. The
weather follows a first-order discrete Markov process with the following
initial state probability and state transition probabilities:

.. code:: python

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
However, instead of observing this process directly, we observe whether
Bob is 'walking', 'shopping' or 'cleaning'. The probability of each
activity depends on the current weather as follows:

.. code:: python

    from bayespy.nodes import Categorical, Mixture
    # Emission probabilities
    P = [[0.1, 0.4, 0.5],
         [0.6, 0.3, 0.1]]
    # Observed process
    Y = Mixture(Z, Categorical, P)
In order to test our method, we'll generate artificial data using this
model:

.. code:: python

    # Draw realization of the weather process
    weather = Z.random()
    # Using this weather, draw realizations of the activities
    activity = Mixture(weather, Categorical, P).random()
Now, using this data, we set our variable :math:`Y` to be observed:

.. code:: python

    Y.observe(activity)
In order to run inference, we construct variational Bayesian inference
engine:

.. code:: python

    from bayespy.inference import VB
    Q = VB(Y, Z)
Note that we need to give all random variables to ``VB``. In this case,
the only random variables were ``Y`` and ``Z``. Next we run the
inference, that is, compute our posterior distribution:

.. code:: python

    Q.update()

.. parsed-literal::

    Iteration 1: loglike=-1.088929e+03 (0.110 seconds)


In this case, because there is only one unobserved random variable, we
recover the exact posterior distribution and there is no need to iterate
more than one step.

Unknown parameters
------------------

Next, we consider the case when we do not know the parameters of the
weather process (initial state probability and state transition
probabilities). We give these parameters quite non-informative priors,
but it is possible to provide more informative priors if such
information is available. First, the weather process:

.. code:: python

    from bayespy.nodes import Dirichlet
    # Initial state probabilities
    a0 = Dirichlet([0.1, 0.1])
    # State transition probabilities
    A = Dirichlet([[0.1, 0.1],
                   [0.1, 0.1]])
    # Markov chain
    Z = CategoricalMarkovChain(a0, A, states=N)
Second, the emission probabilities are also given quite non-informative
priors:

.. code:: python

    # Emission probabilities
    P = Dirichlet([[0.1, 0.1, 0.1],
                   [0.1, 0.1, 0.1]])
    # Observed process
    Y = Mixture(Z, Categorical, P)
We use the same data as before:

.. code:: python

    Y.observe(activity)
Because ``VB`` takes all the unknown variables, we need to provide
``A``, ``a0`` and ``P`` also:

.. code:: python

    Q = VB(Y, Z, A, a0, P)
If we ran the VB algorithm now, we would get a result where all both
states would have identical emission probability distribution. This
happens because of a non-random default initialization. ``P`` is
initialized in such a way that both states have the same distribution,
and ``Z`` is initialized in such a way that each state has equal
probability. Thus, the VB algorithm won't separate them. In such cases,
it is necessary to use a random initialization. In principle, it is
possible to use random initialization for either variable and then
update the other variable first. In the case of mixture distributions,
it might be better to initialize the parameters (``P``) randomly and
update the state assignments (``Z``) first.

.. code:: python

    P.initialize_from_random()
    Q.update(Z, A, a0, P, repeat=20)

.. parsed-literal::

    Iteration 1: loglike=-1.109896e+03 (0.100 seconds)
    Iteration 2: loglike=-1.107351e+03 (0.100 seconds)
    Iteration 3: loglike=-1.106835e+03 (0.100 seconds)
    Iteration 4: loglike=-1.106789e+03 (0.100 seconds)
    Iteration 5: loglike=-1.106778e+03 (0.090 seconds)
    Iteration 6: loglike=-1.106772e+03 (0.100 seconds)
    Iteration 7: loglike=-1.106767e+03 (0.100 seconds)
    Iteration 8: loglike=-1.106763e+03 (0.100 seconds)
    Iteration 9: loglike=-1.106758e+03 (0.100 seconds)
    Iteration 10: loglike=-1.106754e+03 (0.090 seconds)
    Iteration 11: loglike=-1.106750e+03 (0.100 seconds)
    Iteration 12: loglike=-1.106745e+03 (0.100 seconds)
    Iteration 13: loglike=-1.106741e+03 (0.100 seconds)
    Iteration 14: loglike=-1.106736e+03 (0.100 seconds)
    Iteration 15: loglike=-1.106732e+03 (0.100 seconds)
    Iteration 16: loglike=-1.106728e+03 (0.090 seconds)
    Iteration 17: loglike=-1.106724e+03 (0.100 seconds)
    Iteration 18: loglike=-1.106719e+03 (0.100 seconds)
    Iteration 19: loglike=-1.106715e+03 (0.100 seconds)
    Iteration 20: loglike=-1.106711e+03 (0.100 seconds)
    Iteration 21: loglike=-1.106707e+03 (0.100 seconds)
    Iteration 22: loglike=-1.106703e+03 (0.090 seconds)
    Iteration 23: loglike=-1.106699e+03 (0.100 seconds)
    Iteration 24: loglike=-1.106695e+03 (0.100 seconds)
    Iteration 25: loglike=-1.106691e+03 (0.100 seconds)
    Iteration 26: loglike=-1.106687e+03 (0.100 seconds)
    Iteration 27: loglike=-1.106683e+03 (0.100 seconds)
    Iteration 28: loglike=-1.106679e+03 (0.100 seconds)
    Iteration 29: loglike=-1.106675e+03 (0.100 seconds)
    Iteration 30: loglike=-1.106671e+03 (0.090 seconds)
    Iteration 31: loglike=-1.106667e+03 (0.100 seconds)
    Iteration 32: loglike=-1.106663e+03 (0.100 seconds)
    Iteration 33: loglike=-1.106659e+03 (0.090 seconds)
    Iteration 34: loglike=-1.106655e+03 (0.100 seconds)
    Iteration 35: loglike=-1.106651e+03 (0.100 seconds)
    Iteration 36: loglike=-1.106647e+03 (0.100 seconds)
    Iteration 37: loglike=-1.106643e+03 (0.100 seconds)
    Iteration 38: loglike=-1.106639e+03 (0.090 seconds)
    Iteration 39: loglike=-1.106635e+03 (0.100 seconds)
    Iteration 40: loglike=-1.106632e+03 (0.100 seconds)
    Iteration 41: loglike=-1.106628e+03 (0.100 seconds)
    Iteration 42: loglike=-1.106624e+03 (0.090 seconds)
    Iteration 43: loglike=-1.106620e+03 (0.090 seconds)
    Iteration 44: loglike=-1.106616e+03 (0.100 seconds)
    Iteration 45: loglike=-1.106612e+03 (0.100 seconds)
    Iteration 46: loglike=-1.106609e+03 (0.100 seconds)
    Iteration 47: loglike=-1.106605e+03 (0.100 seconds)
    Iteration 48: loglike=-1.106601e+03 (0.090 seconds)
    Iteration 49: loglike=-1.106597e+03 (0.100 seconds)
    Iteration 50: loglike=-1.106593e+03 (0.100 seconds)
    Iteration 51: loglike=-1.106589e+03 (0.100 seconds)
    Iteration 52: loglike=-1.106586e+03 (0.100 seconds)
    Iteration 53: loglike=-1.106582e+03 (0.090 seconds)
    Iteration 54: loglike=-1.106578e+03 (0.090 seconds)
    Iteration 55: loglike=-1.106574e+03 (0.100 seconds)
    Iteration 56: loglike=-1.106570e+03 (0.100 seconds)
    Iteration 57: loglike=-1.106567e+03 (0.100 seconds)
    Iteration 58: loglike=-1.106563e+03 (0.100 seconds)
    Iteration 59: loglike=-1.106559e+03 (0.100 seconds)
    Iteration 60: loglike=-1.106555e+03 (0.100 seconds)
    Iteration 61: loglike=-1.106551e+03 (0.100 seconds)
    Iteration 62: loglike=-1.106547e+03 (0.090 seconds)
    Iteration 63: loglike=-1.106544e+03 (0.100 seconds)
    Iteration 64: loglike=-1.106540e+03 (0.100 seconds)
    Iteration 65: loglike=-1.106536e+03 (0.090 seconds)
    Iteration 66: loglike=-1.106532e+03 (0.100 seconds)
    Iteration 67: loglike=-1.106528e+03 (0.100 seconds)
    Iteration 68: loglike=-1.106524e+03 (0.100 seconds)
    Iteration 69: loglike=-1.106520e+03 (0.100 seconds)
    Iteration 70: loglike=-1.106516e+03 (0.100 seconds)
    Iteration 71: loglike=-1.106512e+03 (0.100 seconds)
    Iteration 72: loglike=-1.106508e+03 (0.100 seconds)
    Iteration 73: loglike=-1.106505e+03 (0.090 seconds)
    Iteration 74: loglike=-1.106501e+03 (0.100 seconds)
    Iteration 75: loglike=-1.106497e+03 (0.100 seconds)
    Iteration 76: loglike=-1.106492e+03 (0.090 seconds)
    Iteration 77: loglike=-1.106488e+03 (0.090 seconds)
    Iteration 78: loglike=-1.106484e+03 (0.100 seconds)
    Iteration 79: loglike=-1.106480e+03 (0.100 seconds)
    Iteration 80: loglike=-1.106476e+03 (0.100 seconds)
    Iteration 81: loglike=-1.106472e+03 (0.100 seconds)
    Iteration 82: loglike=-1.106468e+03 (0.090 seconds)
    Iteration 83: loglike=-1.106464e+03 (0.100 seconds)
    Iteration 84: loglike=-1.106460e+03 (0.090 seconds)
    Iteration 85: loglike=-1.106455e+03 (0.100 seconds)
    Iteration 86: loglike=-1.106451e+03 (0.100 seconds)
    Iteration 87: loglike=-1.106447e+03 (0.100 seconds)
    Iteration 88: loglike=-1.106442e+03 (0.090 seconds)
    Iteration 89: loglike=-1.106438e+03 (0.100 seconds)
    Iteration 90: loglike=-1.106434e+03 (0.100 seconds)
    Iteration 91: loglike=-1.106429e+03 (0.100 seconds)
    Iteration 92: loglike=-1.106425e+03 (0.100 seconds)
    Iteration 93: loglike=-1.106420e+03 (0.090 seconds)
    Iteration 94: loglike=-1.106416e+03 (0.100 seconds)
    Iteration 95: loglike=-1.106411e+03 (0.100 seconds)
    Iteration 96: loglike=-1.106407e+03 (0.100 seconds)
    Iteration 97: loglike=-1.106402e+03 (0.100 seconds)
    Iteration 98: loglike=-1.106397e+03 (0.090 seconds)
    Iteration 99: loglike=-1.106393e+03 (0.100 seconds)
    Iteration 100: loglike=-1.106388e+03 (0.090 seconds)


In order to update the variables in that order, one may explicitly give
the nodes in that order to the ``update`` method. However, the default
update order is the one used when constructing ``Q``, which is the same
in this case, thus we could have ignored listing the nodes to the
``update`` method.

Plot the estimated state transition probabilities:

.. code:: python

    # NOTE: These three lines are just to enable inline plotting in IPython Notebooks.
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.plot([])
    # Plot the state transition matrix
    import bayespy.plot.plotting as bpplt
    bpplt.dirichlet_hinton(A)


.. image:: hmm_discrete_files/hmm_discrete_28_0.png


Plot the estimated emission probabilities:

.. code:: python

    bpplt.dirichlet_hinton(P)


.. image:: hmm_discrete_files/hmm_discrete_30_0.png


It is interesting that these estimated parameters are very different
from the true parameters. This happens because of un-identifiability:
