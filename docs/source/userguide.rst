User's guide
============

* Construct the model (Bayesian network)

* Put the data in

* Run inference

* Examine posterior results

Simple example
--------------

.. image:: images/model01.*

First, the Bayesian

.. literalinclude:: examples/example_01.py
   :start-after: (1)
   :end-before: (2)


Constructing the model
----------------------

The model is constructed as a Bayesian network, which is directed
acyclic graph representing a set of random variables and their
conditional dependencies.



Performing inference
--------------------

First, generate some data:

.. literalinclude:: examples/example_01.py
   :start-after: (2)
   :end-before: (3)

Run the inference

.. literalinclude:: examples/example_01.py
   :start-after: (3)
   :end-before: (4)

Show the resulting posterior approximation

.. literalinclude:: examples/example_01.py
   :start-after: (4)

Plates


