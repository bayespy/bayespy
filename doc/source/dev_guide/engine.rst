..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Implementing inference engines
==============================

Currently, only variational Bayesian inference engine is implemented.  This
implementation is not very modular, that is, the inference engine is not well
separated from the model construction.  Thus, it is not straightforward to
implement other inference engines at the moment.  Improving the modularity of
the inference engine and model construction is future work with high priority.
In any case, BayesPy aims to be an efficient, simple and modular Bayesian
package for variational inference at least.
