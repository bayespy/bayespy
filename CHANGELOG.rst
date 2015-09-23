Version 0.3.7 (2015-09-23)
++++++++++++++++++++++++++

 * Enable keyword arguments when plotting via the inference engine

 * Add initial support for logging

Version 0.3.6 (2015-08-12)
++++++++++++++++++++++++++

 * Add maximum likelihood node for the shape parameter of Gamma

 * Fix Hinton diagrams for 1-D and 0-D Gaussians

 * Fix autosave interval counter

 * Fix bugs in constant nodes

Version 0.3.5 (2015-06-09)
++++++++++++++++++++++++++

 * Fix indexing bug in VB optimization (not VB-EM)

 * Fix demos

Version 0.3.4 (2015-06-09)
++++++++++++++++++++++++++

 * Fix computation of probability density of Dirichlet nodes

 * Use unit tests for all code snippets in docstrings and documentation

Version 0.3.3 (2015-06-05)
++++++++++++++++++++++++++

 * Change license to the MIT license

 * Improve SumMultiply efficiency

 * Hinton diagrams for gamma variables

 * Possible to load only nodes from HDF5 results

Version 0.3.2 (2015-03-16)
++++++++++++++++++++++++++

 * Concatenate node added

 * Unit tests for plotting fixed

Version 0.3.1 (2015-03-12)
++++++++++++++++++++++++++

 * Gaussian mixture 2D plotting improvements

 * Covariance matrix sampling improvements

 * Minor documentation fixes

Version 0.3 (2015-03-05)
++++++++++++++++++++++++

 * Add gradient-based optimization methods (Riemannian/natural gradient or normal)

 * Add collapsed inference

 * Add the pattern search method

 * Add deterministic annealing

 * Add stochastic variational inference

 * Add optional input signals to Gaussian Markov chains

 * Add unit tests for plotting functions (by Hannu Hartikainen)

 * Add printing support to nodes

 * Drop Python 3.2 support

Version 0.2.3 (2014-12-03)
++++++++++++++++++++++++++

 * Fix matplotlib compatibility broken by recent changes in matplotlib

 * Add random sampling for Binomial and Bernoulli nodes

 * Fix minor bugs, for instance, in plot module

Version 0.2.2 (2014-11-01)
++++++++++++++++++++++++++

 * Fix normalization of categorical Markov chain probabilities (fixes HMM demo)

 * Fix initialization from parameter values

Version 0.2.1 (2014-09-30)
++++++++++++++++++++++++++

 * Add workaround for matplotlib 1.4.0 bug related to interactive mode which
   affected monitoring

 * Fix bugs in Hinton diagrams for Gaussian variables

Version 0.2 (2014-08-06)
++++++++++++++++++++++++

 * Added all remaining common distributions: Bernoulli, binomial, multinomial,
   Poisson, beta, exponential.

 * Added Gaussian arrays (not just scalars or vectors).

 * Added Gaussian Markov chains with time-varying or swithing dynamics.

 * Added discrete Markov chains (enabling hidden Markov models).

 * Added joint Gaussian-Wishart and Gaussian-gamma nodes.
 
 * Added deterministic gating node.

 * Added deterministic general sum-product node.

 * Added parameter expansion for Gaussian arrays and time-varying/switching
   Gaussian Markov chains.

 * Added new plotting functions: pdf, Hinton diagram.

 * Added monitoring of posterior distributions during iteration.

 * Finished documentation and added API.

Version 0.1 (2013-07-25)
++++++++++++++++++++++++

 * Added variational message passing inference engine.
 
 * Added the following common distributions: Gaussian vector, gamma, Wishart,
   Dirichlet, categorical.

 * Added Gaussian Markov chain.

 * Added parameter expansion for Gaussian vectors and Gaussian Markov chain.

 * Added stochastic mixture node.

 * Added deterministic dot product node.

 * Created preliminary version of the documentation.

