
                
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

    # This is the PCA model from the previous section
    import numpy as np
    from bayespy.nodes import GaussianARD, Gamma, Dot
    D = 3
    X = GaussianARD(0, 1,
                    shape=(D,),
                    plates=(1,100),
                    name='X')
    alpha = Gamma(1e-3, 1e-3,
                  plates=(D,),
                  name='alpha')
    C = GaussianARD(0, alpha,
                    shape=(D,),
                    plates=(10,1),
                    name='C')
    F = Dot(C, X)
    tau = Gamma(1e-3, 1e-3)
    Y = GaussianARD(F, tau)
                
                
Performing inference
====================

Approximation of the posterior distribution can be divided into several steps:

-  Observe some nodes

-  Choose the inference engine

-  Initialize the posterior approximation

-  Run the inference algorithm

In order to illustrate these steps, we'll be using the PCA model constructed in
the previous section.

Observing nodes
---------------

First, let us generate some toy data:

>>> c = np.random.randn(10, 2)
>>> x = np.random.randn(2, 100)
>>> data = np.dot(c, x) + 0.1*np.random.randn()

The data is provided by simply calling ``observe`` method of a stochastic node:

>>> Y.observe(data)
                
It is important that the shape of the ``data`` array matches the plates and
shape of the node ``y`` For instance, if ``y`` is :class:`Wishart` node for
:math:`3\times 3` matrices with plates ``(5,1,10)``, the full shape of ``y``
would be ``(5,1,10,3,3)``.  The data array must have this shape exactly, that
is, no broadcasting rules are applied.
                
Missing values
~~~~~~~~~~~~~~

It is possible to mark missing values by providing a mask:

.. code:: python

    y.observe(data, mask=[True, False, False, True, True,
                          False, True, True, True, False])
``True`` means that the value is observed and ``False`` means that the
value is missing. The mask is applied to the *plates*, not to the data
array directly. This means that it is not possible to observe a random
variable partially, each repetition defined by the plates is either
fully observed or fully missing. Thus, the mask is applied to the
plates. It is often possible to circumvent this seemingly tight
restriction by adding an observable child node which factorizes more.

The shape of the mask is broadcasted to plates using standard NumPy
broadcasting rules. So, if the variable has plates ``(5,1,10)``, the
mask could have a shape ``()``, ``(1,)``, ``(1,1)``, ``(1,1,1)``,
``(10,)``, ``(1,10)``, ``(1,1,10)``, ``(5,1,1)`` or ``(5,1,10)``. In
order to speed up the inference, missing plates are automatically
ignored by the inference algorithm if they are not needed. Thus, the
missing values are integrated out giving more accurate approximations
faster.

Choosing the inference method
-----------------------------

                
Inference methods can be found in :mod:`bayespy.inference` package.
Currently, only variational Bayesian approximation is implemented
(:class:`bayespy.inference.VB`).  The inference engine is constructed by
giving the nodes of the model.
                
.. code:: python

    from bayespy.inference import VB
    Q = VB(node1, node2, node3, node4)
Initializing the inference
--------------------------

The inference engines give some initialization to the nodes by default.
However, the inference algorithms can be sensitive to the
initialization, thus it is sometimes necessary to have full control over
the initialization. There may be different initialization methods, but
for VB you can, for instance, initialize in one of the following ways:

-  ``initialize_from_prior``: Use only parent nodes to update the node.

-  ``initialize_from_parameters``: Use the given parameter values for
   the distribution.

A random initialization for VB has to be performed manually, because it
is not obvious what is actually wanted. For instance, one way to achieve
it is to first update from the parents, then to draw a random sample
from that distribution and to set the values of the parameters based on
that. For ``Normal`` node, one could draw the mean parameter randomly
and choose the precision parameter arbitrarily:

.. code:: python

    x = bp.nodes.Normal(mu, tau, plates=(10,))
    x.initialize_from_prior()
    x.initialize_from_parameters(x.random(), 1)
In this case, the precision was set to one. The default initialization
method is ``initialization_from_prior``, which is performed when the
node is created. If the initialization uses the values of the parents,
they should be initialized before the children.

Running the inference algorithm
-------------------------------

The approximation methods are based on iterative algorithms, which can
be run using ``update`` method. By default, it takes one iteration step
updating all nodes once. However, you can give as arguments the nodes
you want to update and they are updated in the given order. It is
possible to give same nodes several times, for instance:

.. code:: python

    Q.update(node1, node3, node1, node4)
This would update ``node3`` and ``node4`` once, and ``node1`` twice. In
order to update several times, one can use the optional argument
``repeat``.

.. code:: python

    Q.update(node3, node4, repeat=5)
    Q.update(node1, node2, node3, node4, repeat=10)
This first updates ``node3`` and ``node4`` five times and then all the
nodes ten times. This might be useful, for instance, if updating some
nodes is expensive and should be done rarely or if updating some nodes
in the beginning would cause the algorithm to converge to a bad
solution.

                
.. warning::

   Ideally, one constructs the model and then chooses the inference
   method to be used - possibly trying several different methods.
   However, the model construction is not yet separated from the model
   construction, that is, the constructed model network is also the
   variational message passing network for VB inference.
                
