..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


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


Moments
-------

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


When implementing a new node, it is not always necessary to implement a new
moments class.  If another node has the same sufficient statistic vector, thus
the same moments, that moments class can be used.  Otherwise, one must implement
a simple moments class which has the following methods:

 * :func:`Moments.compute_fixed_moments`

      Computes the moments for a known value.  This is used to compute the
      moments of constant numeric arrays and wrap them into constant nodes.

 * :func:`Moments.compute_dims_from_values`

      Given a known value of the variable, return the shape of the variable
      dimensions in the moments.  This is used to solve the shape of the moments
      array for constant nodes.


Distributions
-------------

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
node class.  First, we need to explain a few important attributes before we can
explain how to implement a node class.

Stochastic exponential family nodes have two attributes that store the state of
the posterior distribution:

 * ``phi``

   The natural parameter vector :math:`\tilde{\boldsymbol{\phi}}` of the
   posterior approximation.

 * ``u``

   The moments :math:`\langle \mathbf{u} \rangle` of the posterior
   approximation.

Instead of storing these two variables as vectors (as in the mathematical
formulas), they are stored as lists of arrays with convenient shapes.  For
instance, :class:`Gaussian` node stores the moments as a list consisting of a
vector :math:`\langle \mathbf{x} \rangle` and a matrix :math:`\langle
\mathbf{xx}^T \rangle` instead of reshaping and concatenating these into a
single vector.  The same applies for the natural parameters ``phi`` because it
has the same shape as ``u``.


The shapes of the arrays in the lists ``u`` and ``phi`` consist of the shape
caused by the plates and the shape caused by the variable itself.  For instance,
the moments of :class:`Gaussian` node have shape ``(D,)`` and ``(D, D)``, where
``D`` is the dimensionality of the Gaussian vector.  In addition, if the node
has plates, they are added to these shapes.  Thus, for instance, if the
:class:`Gaussian` node has plates ``(3, 7)`` and ``D`` is 5, the shape of
``u[0]`` and ``phi[0]`` would be ``(3, 7, 5)`` and the shape of ``u[1]`` and
``phi[1]`` would be ``(3, 7, 5, 5)``.  This shape information is stored in the
following attributes:

 * ``plates`` : a tuple

   The plates of the node.  In our example, ``(3, 7)``.

 * ``dims`` : a list of tuples

   The shape of each of the moments arrays (or natural parameter arrays) without
   plates.  In our example, ``[ (5,), (5, 5) ]``.


Finally, three attributes define VMP for the node:

 * ``_moments`` : :class:`Moments` sub-class instance

   An object defining the moments of the node.

 * ``_parent_moments`` : list of :class:`Moments` sub-class instances

   A list defining the moments requirements for each parent.

 * ``_distribution`` : :class:`Distribution` sub-class instance

   An object implementing the VMP formulas.

Basically, a node class is a collection of the above attributes.  When a node is
created, these attributes are defined.  The base class for exponential family
nodes, :class:`ExponentialFamily`, provides a simple default constructor which
does not need to be overwritten if ``dims``, ``_moments``, ``_parent_moments``
and ``_distribution`` can be provided as static class attributes.  For instance,
:class:`Gamma` node defines these attributes statically.  However, usually at
least one of these attributes cannot be defined statically in the class.  In
that case, one must implement a class method which overloads
:func:`ExponentialFamily._constructor`.  The purpose of this method is to define
all the attributes given the parent nodes.  These are defined using a class
method instead of ``__init__`` method in order to be able to use the class
constructors statically, for instance, in :class:`Mixture` class.  This
construction allows users to create mixtures of any exponential family
distribution with simple syntax.


The parents of a node must be converted so that they have a correct message
type, because the user may have provided numeric arrays or nodes with incorrect
message type.  Numeric arrays should be converted to constant nodes with correct
message type.  Incorrect message type nodes should be converted to correct
message type nodes if possible.  Thus, the constructor should use
``Node._ensure_moments`` method to make sure the parent is a node with correct
message type.  Instead of calling this method for each parent node in the
constructor, one can use ``ensureparents`` decorator to do this automatically.
However, the decorator requires that ``_parent_moments`` attribute has already
been defined statically.  If this is not possible, the parent nodes must be
converted manually in the constructor, because one should never assume that the
parent nodes given to the constructor are nodes with correct message type or
even nodes at all.




Deterministic nodes
-------------------


Deterministic nodes are nodes that do not correspond to any probability
distribution but rather a deterministic function.  It does not have any moments
or natural parameters to store.  A deterministic node is implemented as a
subclass of :class:`Deterministic` base class.  The new node class must
implement the following methods:

 * :func:`Deterministic._compute_moments`

   Computes the moments given the moments of the parents.

 * :func:`Deterministic._compute_message_to_parent`

   Computes the message to a parent node given the message from the children and
   the moments of the other parents.  In some cases, one may want to implement
   :func:`Deterministic._compute_message_and_mask_to_parent` or
   :func:`Deterministic._message_to_parent` instead in order to gain more
   control over efficient computation.


Similarly as in :class:`Distribution` class, if the node handles plates
irregularly, it is important to implement the following methods:

 * :func:`Deterministic._plates_from_parent`

   Given the plates of the parent, return the resulting plates of the child.

 * :func:`Deterministic._plates_to_parent`

   Given the plates of the child, return the plates of the parent that would
   have resulted them.

 * :func:`Deterministic._compute_weights_to_parent`

   Given the mask array, convert it to a plate mask of the parent.


 
Converter nodes
+++++++++++++++

Sometimes a node has incorrect message type but the message can be converted
into a correct type.  For instance, :class:`GaussianMarkovChain` has
:class:`GaussianMarkovChainMoments` message type, which means moments :math:`\{
\langle \mathbf{x}_n \rangle, \langle \mathbf{x}_n \mathbf{x}_n^T \rangle,
\langle \mathbf{x}_n \mathbf{x}_{n-1}^T \rangle \}^N_{n=1}`.  These moments can
be converted to :class:`GaussianMoments` by ignoring the third element and
considering the time axis as a plate axis.  Thus, if a node requires
:class:`GaussianMoments` message from its parent, :class:`GaussianMarkovChain`
is a valid parent if its messages are modified properly.  This conversion is
implemented in :class:`_MarkovChainToGaussian` converter class.  Converter nodes
are simple deterministic nodes that have one parent node and they convert the
messages to another message type.


For the user, it is not convenient if the exact message type has to be known and
an explicit converter node needs to be created.  Thus, the conversions are done
automatically and the user will be unaware of them.  In order to enable this
automatization, when writing a converter node, one should register the converter
to the moments class using :func:`Moments.add_converter`.  For instance, a class
``X`` which converts moments ``A`` to moments ``B`` is registered as
``A.add_conveter(B, X)``.  After that, :func:`Node._ensure_moments` and
:func:`Node._convert` methods are used to perform the conversions automatically.
The conversion can consist of several consecutive converter nodes, and the least
number of conversions is used.
