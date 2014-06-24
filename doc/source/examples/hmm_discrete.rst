
Discrete hidden Markov model
============================

This example is also available as `an IPython
notebook <hmm_discrete.ipynb>`__ or `a Python
script <hmm_discrete.py>`__.

Known parameters
----------------

This example follows the one presented in
`Wikipedia <http://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example>`__.
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

    Iteration 1: loglike=-1.091583e+03 (0.090 seconds)


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

    Iteration 1: loglike=-1.115941e+03 (0.090 seconds)
    Iteration 2: loglike=-1.115671e+03 (0.090 seconds)
    Iteration 3: loglike=-1.115603e+03 (0.100 seconds)
    Iteration 4: loglike=-1.115574e+03 (0.090 seconds)
    Iteration 5: loglike=-1.115555e+03 (0.090 seconds)
    Iteration 6: loglike=-1.115538e+03 (0.100 seconds)
    Iteration 7: loglike=-1.115521e+03 (0.090 seconds)
    Iteration 8: loglike=-1.115504e+03 (0.090 seconds)
    Iteration 9: loglike=-1.115487e+03 (0.090 seconds)
    Iteration 10: loglike=-1.115469e+03 (0.090 seconds)
    Iteration 11: loglike=-1.115451e+03 (0.100 seconds)
    Iteration 12: loglike=-1.115433e+03 (0.090 seconds)
    Iteration 13: loglike=-1.115413e+03 (0.090 seconds)
    Iteration 14: loglike=-1.115394e+03 (0.090 seconds)
    Iteration 15: loglike=-1.115374e+03 (0.090 seconds)
    Iteration 16: loglike=-1.115354e+03 (0.100 seconds)
    Iteration 17: loglike=-1.115333e+03 (0.090 seconds)
    Iteration 18: loglike=-1.115312e+03 (0.090 seconds)
    Iteration 19: loglike=-1.115290e+03 (0.090 seconds)
    Iteration 20: loglike=-1.115268e+03 (0.090 seconds)


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


.. image:: hmm_discrete_files/hmm_discrete_29_0.png


Plot the estimated emission probabilities:

.. code:: python

    bpplt.dirichlet_hinton(P)


.. image:: hmm_discrete_files/hmm_discrete_31_0.png


It is interesting that these estimated parameters are very different
from the true parameters. This happens because of un-identifiability:
different parameters lead to similar marginal distributions over the
observed process.
