User's guide
============

* Construct the model (Bayesian network)

* Put the data in

* Run inference

* Examine posterior results

Simple example
--------------

Let's begin with a simple example which shows the basic steps.  In
this case, we do not use any real data but generate some toy data.
The dataset consists of ten samples from the Gaussian distribution
with mean 5 and standard deviation 10:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (0)
   :end-before: (1)

Now, given this data we would like to estimate the mean and the
standard deviation. We can construct a simple model shown below as a
directed factor graph.

.. bayesnet:: Directed factor graph of the example model.

   \node[obs]                                  (y)     {$y$} ;
   \node[latent, above left=1.5 and 0.5 of y]  (mu)    {$\mu$} ;
   \node[latent, above right=1.5 and 0.5 of y] (tau)   {$\tau$} ;
   \node[const, above=of mu, xshift=-0.5cm]    (mumu)  {$0$} ;
   \node[const, above=of mu, xshift=0.5cm]     (taumu) {$10^{-3}$} ;
   \node[const, above=of tau, xshift=-0.5cm]   (atau)  {$10^{-3}$} ;
   \node[const, above=of tau, xshift=0.5cm]    (btau)  {$10^{-3}$} ;

   \factor[above=of y] {y-f} {left:$\mathcal{N}$} {mu,tau}     {y};
   \factor[above=of mu] {} {left:$\mathcal{N}$}   {mumu,taumu} {mu};
   \factor[above=of tau] {} {left:$\mathcal{G}$}  {atau,btau}  {tau};

   \plate {} {(y)(y-f)(y-f-caption)} {10} ;

Alternatively, the model can also be defined using explicit
mathematical notation:

.. math::

   p(\mathbf{y}|\mu,\tau) &= \prod^{10}_{n=1} \mathcal{N}(y_n|\mu,\tau) \\
   p(\mu) &= \mathcal{N}(\mu|0,10^{-3}) \\
   p(\tau) &= \mathcal{G}(\tau|10^{-3},10^{-3})

This model can be defined in BayesPy as follows:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (1)
   :end-before: (2)

This is quite self-explanatory given the model definitions above.
Now, we need to use the generated data:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (2)
   :end-before: (3)

Next we want to estimate the posterior distribution.  In principle, we
could use different inference engines (e.g., MCMC or EP) but currently
only variational Bayesian (VB) engine is implemented.  The engine is
initialized by giving the nodes and the the inference algorithm can be
run as long as wanted (20 iterations in this case):

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (3)
   :end-before: (4)

In VB, the true posterior :math:`p(\mu,\tau|\mathbf{y})` is
approximated with a factorized distribution :math:`q(\mu)q(\tau)`.
The resulting approximate posterior distributions :math:`q(\mu)` and
:math:`q(\tau)` can be examined as:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (4)

This example was a very simple introduction to using BayesPy.  The
model can be much more complex and each phase contains more options to
give the user more control over the inference.  The following sections
give more details.


Constructing the model
----------------------

The model is constructed as a Bayesian network, which is directed
acyclic graph representing a set of random variables and their
conditional dependencies.

Plates

Providing the data
------------------


Performing inference
--------------------


Examining the results
---------------------
