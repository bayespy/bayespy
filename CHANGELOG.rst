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

