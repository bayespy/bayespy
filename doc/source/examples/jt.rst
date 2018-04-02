..
   Copyright (C) 2018 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


.. testsetup::

   import numpy
   numpy.random.seed(1)

Junction tree algorithm
=======================

This example shows how to perform exact inference on discrete graphs by using
the junction tree algorithm. These graphs can also be a part of a larger
graphical model in which case approximate VB inference is performed for the
entire model.

>>> import bayespy as bp
>>> X = bp.nodes.CategoricalGraph(
...     {
...         "choose_dice": {
...             "table": [0.5, 0.5],
...         },
...         "throw_dice": {
...             "given": ["choose_dice"],
...             "plates": ["trials"],
...             "table": [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]],
...         },
...     },
...     plates={
...         "trials": 10
...     },
... )
>>> X.observe({"throw_dice": [0, 5, 3, 5, 1, 5, 5, 0, 5, 5]})
>>> X.update()
>>> X.get_moments()["choose_dice"]


Observation probability distributions
-------------------------------------

TODO

(E.g., Gaussian)


Learn parameters
----------------

TODO

(With Dirichlet priors)
