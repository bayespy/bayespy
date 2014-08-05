..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public License.
   See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License version 3 as published by the Free
   Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
   A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with
   BayesPy.  If not, see <http://www.gnu.org/licenses/>.


Implementing nodes
==================

The main goal of BayesPy is to provide a package which enables easy and flexible
construction of simple and complex models with efficient inference.  However,
users may sometimes be unable to construct their models because the built-in
nodes do not implement some specific features.  Thus, one may need to implement
new nodes in order to construct the model.  BayesPy aims to make the
implementation of new nodes both simple and fast.  Probably, a large complex
model can be constructed almost completely with the built-in nodes and the user
needs to implement only a few nodes.


Messaging framework
-------------------

.. currentmodule:: bayespy.inference.vmp.nodes.node

In order to implement nodes, it is important to understand the messaging
framework of the nodes.  A node is a unit of calculation which communicates to
its parent and child nodes using messages.  These messages have types that need
to match between nodes, that is, the child node needs to understand the messages
its parents are sending and vice versa.  Thus, a node defines which message type
it requires from each of its parents, and only nodes that have that type of
output message (i.e., the message to a child node) are valid parent nodes for
that node.


The message type is defined by the moments of the parent node.  The moments are
a collection of expectations: :math:`\{ \langle f_1(X) \rangle, \ldots, \langle
f_N(X) \rangle \}`.  The functions :math:`f_1, \ldots, f_N` (and the number of
the functions) define the message type and they are the sufficient statistic as
discussed in the previous section.  Different message types are represented by
:class:`Moments` class hierarchy.  For instance, :class:`GaussianMoments`
represents a message type with parent moments :math:`\{\langle \mathbf{x}
\rangle, \langle \mathbf{xx}^T \rangle \}` and :class:`WishartMoments` a message
type with parent moments :math:`\{\langle \mathbf{\Lambda} \rangle, \langle \log
|\mathbf{\Lambda}| \rangle\}`.


.. currentmodule:: bayespy.nodes

Let us give an example: :class:`Gaussian` node outputs :class:`GaussianMoments`
messages and :class:`Wishart` node outputs :class:`WishartMoments` messages.
:class:`Gaussian` node requires that it receives :class:`GaussianMoments`
messages from the mean parent node and :class:`WishartMoments` messages from the
precision parent node.  Thus, :class:`Gaussian` and :class:`Wishart` are valid
node classes as the mean and precision parent nodes of :class:`Gaussian` node.


Note that several nodes may have the same output message type and some message
types can be transformed to other message types using deterministic converter
nodes.  For instance, :class:`Gaussian` and :class:`GaussianARD` nodes both
output :class:`GaussianMoments` messages, deterministic :class:`SumMultiply`
also outputs :class:`GaussianMoments` messages, and deterministic converter
:class:`_MarkovChainToGaussian` converts :class:`GaussianMarkovChainMoments` to
:class:`GaussianMoments`.


.. currentmodule:: bayespy.inference.vmp.nodes.node


Each node specifies the message type requirements of its parents by
:attr:`Node._parent_moments` attribute which is a list of :class:`Moments`
sub-class instances.  These moments objects have a few purpose when creating the
node: 1) check that parents are sending proper messages; 2) if parents use
different message type, try to add a converter which converts the messages to
the correct type if possible; 3) if given parents are not nodes but numeric
arrays, convert them to constant nodes with correct output message type.


When implementing a new node, it is not always necessary to implement a moments
class.  If another node has the same sufficient statistic vector, thus the same
moments, that class can be used.  Otherwise, one must implement a simple moments
class which has the following methods:

 * :func:`Moments.compute_fixed_moments`

      Computes the moments for a known value.  This is used to compute the
      moments of constant numeric arrays and wrap them into constant nodes.

 * :func:`Moments.compute_dims_from_values`

      Given a known value of the variable, return the shape of the variable
      dimensions in the moments.  This is used to solve the shape of the moments
      array for constant nodes.


Stochastic distributions
------------------------

.. currentmodule:: bayespy.inference.vmp.stochastic

In order to implement a stochastic exponential family node, one must first write
down the log probability density function of the node and derive the terms
discussed in section :ref:`sec-vmp-terms`.  These terms are implemented and
collected as a class which is a subclass of :class:`Distribution`.  The main
reason to implement these methods in another class instead of the node class
itself is that these methods can be used without creating a node, for instance,
in :class:`Mixture` class.

.. currentmodule:: bayespy.inference.vmp.nodes.expfamily

For exponential family distributions, the distribution class is a subclass of
:class:`ExponentialFamilyDistribution`, and the relation between the terms in
section :ref:`sec-vmp-terms` and the methods is as follows:

 * :func:`ExponentialFamilyDistribution.compute_phi_from_parents`

      Computes the expectation of the natural parameters :math:`\langle
      \boldsymbol{\phi} \rangle` in the prior distribution given the moments of
      the parents.

 * :func:`ExponentialFamilyDistribution.compute_cgf_from_parents`

      Computes the expectation of the negative log normalizer :math:`\langle g
      \rangle` of the prior distribution given the moments of the parents.

 * :func:`ExponentialFamilyDistribution.compute_moments_and_cgf`

      Computes the moments :math:`\langle \mathbf{u} \rangle` and the negative
      log normalizer :math:`\tilde{g}` of the posterior distribution
      given the natural parameters :math:`\tilde{\boldsymbol{\phi}}`.

 * :func:`ExponentialFamilyDistribution.compute_message_to_parent`

      Computes the message :math:`\langle
      \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}} \rangle` from
      the node :math:`\mathbf{x}` to its parent node :math:`\boldsymbol{\theta}`
      given the moments of the node and the other parents.

 * :func:`ExponentialFamilyDistribution.compute_fixed_moments_and_f`

      Computes :math:`\mathbf{u}(\mathbf{x})` and :math:`f(\mathbf{x})` for
      given observed value :math:`\mathbf{x}`.  Without this method, variables
      from this distribution cannot be observed.

For each stochastic exponential family node, one must write a distribution class
which implements these methods.  After that, the node class is basically a
simple wrapper and it also stores the moments and the natural parameters of the
current posterior approximation.  Note that the distribution classes do not
store node-specific information, they are more like static collections of
methods.  However, sometimes the implementations depend on some information,
such as the dimensionality of the variable, and this information must be
provided, if needed, when constructing the distribution object.


In addition to the methods listed above, it is necessary to implement a few more
methods in some cases.  This happens when the plates of the parent do not map to
the plates directly as discussed in section :ref:`sec-irregular-plates`.  Then,
one must write methods that implement this plate mapping and apply the same
mapping to the mask array:

 * :func:`ExponentialFamilyDistribution.plates_from_parent`

      Given the plates of the parent, return the resulting plates of the child.

 * :func:`ExponentialFamilyDistribution.plates_to_parent`

      Given the plates of the child, return the plates of the parent that would
      have resulted them.

 * :func:`ExponentialFamilyDistribution.compute_mask_to_parent`

     Given the mask array of the child, apply the plate mapping.

It is important to understand when one must implement these methods, because the
default implementations in the base class will lead to errors or weird results.


Stochastic exponential family nodes
-----------------------------------

After implementing the distribution class, the next task is to implement the
node class.

 * constructor

 * dims, plates


Deterministic nodes
-------------------


Roughly said, a deterministic node takes the message(s) of its parent(s) and
transforms them to an output message.  The transformation may correspond to some
deterministic function or it may only be a simple modification of the message to
another type.

 * deterministic, stochastic nodes

 * message protocol, input output

 * derive equations


 
Converter nodes
+++++++++++++++
