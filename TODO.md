
* The mask of the observations (missing values) doesn't get propagated until
  update methods are called. This can cause errors in the calculation of the
  lower bound because the mask might change between iterations. Thus, the mask
  should be propagated to parents whenever observed is called. And it should be
  called before starting the VB iteration.

* Is it really necessary or useful to have phi in exponential family nodes?
  Could you just directly update moments and cost? Why to use phi? Maybe it is
  necessary for general distributions such as Mixture?