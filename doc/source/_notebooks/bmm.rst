
Bernoulli mixture model
=======================

blaa blaa blaa

.. code:: python

    from bayespy.nodes import (Mixture,
                                Bernoulli,
                                Beta,
                                Categorical,
                                Dirichlet)
    q = Dirichlet(K*[1e-3],
                  name='q')
    Z = Categorical(q,
                    plates=(N,1),
                    name='Z')
    
    p = Beta([1e-3, 1e-3],
             plates=(D,K),
             name='p')
    X = Mixture(Z, Bernoulli, p)
    
    X.observe(data)
    
    Q = VB(Z, q, X, p)
    p.initialize_from_random()
    
    Q.update(repeat=10)