
Gaussian mixture model
======================

Do some stuff:

.. code:: python

    from bayespy.nodes import Dirichlet
    alpha = Dirichlet([1e-3, 1e-3, 1e-3])
    print(alpha._message_to_child())

.. parsed-literal::

    [array([-666.66994695, -666.66994695, -666.66994695])]


Nice!
