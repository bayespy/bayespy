..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Quick start guide
=================

This short guide shows the key steps in using BayesPy for variational
Bayesian inference by applying BayesPy to a simple problem. The key
steps in using BayesPy are the following:

-  Construct the model

-  Observe some of the variables by providing the data in a proper
   format

-  Run variational Bayesian inference

-  Examine the resulting posterior approximation

To demonstrate BayesPy, we'll consider a very simple problem: we have a
set of observations from a Gaussian distribution with unknown mean and
variance, and we want to learn these parameters. In this case, we do not
use any real-world data but generate some artificial data. The dataset
consists of ten samples from a Gaussian distribution with mean 5 and
standard deviation 10. This dataset can be generated with NumPy as
follows:

.. code:: python

    import numpy as np
    data = np.random.normal(5, 10, size=(10,))
Now, given this data we would like to estimate the mean and the standard
deviation as if we didn't know their values. The model can be defined as
follows:

.. math::


   \begin{split}
   p(\mathbf{y}|\mu,\tau) &= \prod^{9}_{n=0} \mathcal{N}(y_n|\mu,\tau) \\
   p(\mu) &= \mathcal{N}(\mu|0,10^{-6}) \\
   p(\tau) &= \mathcal{G}(\tau|10^{-6},10^{-6})
   \end{split}

where :math:`\mathcal{N}` is the Gaussian distribution parameterized by
its mean and precision (i.e., inverse variance), and :math:`\mathcal{G}`
is the gamma distribution parameterized by its shape and rate
parameters. Note that we have given quite uninformative priors for the
variables :math:`\mu` and :math:`\tau`\ . This simple model can also be
shown as a directed factor graph:

                
.. bayesnet:: Directed factor graph of the example model.

   \node[obs]                                  (y)     {$y_n$} ;
   \node[latent, above left=1.5 and 0.5 of y]  (mu)    {$\mu$} ;
   \node[latent, above right=1.5 and 0.5 of y] (tau)   {$\tau$} ;
   \factor[above=of mu] {mu-f} {left:$\mathcal{N}$} {} {mu} ;
   \factor[above=of tau] {tau-f} {left:$\mathcal{G}$} {} {tau} ;

   \factor[above=of y] {y-f} {left:$\mathcal{N}$} {mu,tau}     {y};

   \plate {} {(y)(y-f)(y-f-caption)} {$n=0,\ldots,9$} ;
                
This model can be constructed in BayesPy as follows:

.. code:: python

    from bayespy.nodes import GaussianARD, Gamma
    mu = GaussianARD(0, 1e-6)
    tau = Gamma(1e-6, 1e-6)
    y = GaussianARD(mu, tau, plates=(10,))
                
.. currentmodule:: bayespy.nodes

This is quite self-explanatory given the model definitions above. We have used two types of nodes :class:`GaussianARD` and :class:`Gamma` to represent Gaussian and gamma distributions, respectively. There are much more distributions in :mod:`bayespy.nodes` so you can construct quite complex conjugate exponential family models. The node :code:`y` uses keyword argument :code:`plates` to define the plates :math:`n=0,\ldots,9`.

Now that we have created the model, we can provide our data by setting :code:`y` as observed:
                
.. code:: python

    y.observe(data)
Next we want to estimate the posterior distribution. In principle, we
could use different inference engines (e.g., MCMC or EP) but currently
only variational Bayesian (VB) engine is implemented. The engine is
initialized by giving all the nodes of the model:

.. code:: python

    from bayespy.inference import VB
    Q = VB(mu, tau, y)
The inference algorithm can be run as long as wanted (max. 20 iterations
in this case):

.. code:: python

    Q.update(repeat=20)

.. parsed-literal::

    Iteration 1: loglike=-5.789562e+01 (0.010 seconds)
    Iteration 2: loglike=-5.612083e+01 (0.000 seconds)
    Iteration 3: loglike=-5.611848e+01 (0.010 seconds)
    Iteration 4: loglike=-5.611846e+01 (0.000 seconds)
    Converged.


Now the algorithm converged after four iterations, before the requested
20 iterations.

VB approximates the true posterior :math:`p(\mu,\tau|\mathbf{y})` with a
distribution which factorizes with respect to the nodes:
:math:`q(\mu)q(\tau)`\ . The resulting approximate posterior
distributions :math:`q(\mu)` and :math:`q(\tau)` can be examined, for
instance, by plotting the marginal probability density functions:

.. code:: python

    import bayespy.plot as bpplt
    # The following two two lines are just for enabling matplotlib plotting in notebooks
    %matplotlib inline
    bpplt.pyplot.plot([])
    bpplt.pyplot.subplot(2, 1, 1)
    bpplt.pdf(mu, np.linspace(-10, 20, num=100), color='k', name=r'\mu')
    bpplt.pyplot.subplot(2, 1, 2)
    bpplt.pdf(tau, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\tau');


.. image:: quickstartbackup_files/quickstartbackup_14_0.png


This example was a very simple introduction to using BayesPy. The model
can be much more complex and each phase contains more options to give
the user more control over the inference. The following sections give
more details about the phases.
