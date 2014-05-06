
Example: Principal component analysis
=====================================

Yeah.

                
.. bayesnet:: Directed factor graph of the example model.

   \node[obs]                                  (y)     {$y$} ;
   \node[latent, above left=1.5 and 0.5 of y]  (mu)    {$\mu$} ;
   \node[latent, above right=1.5 and 0.5 of y] (tau)   {$\tau$} ;
   \node[const, above=of mu, xshift=-0.5cm]    (mumu)  {$0$} ;
   \node[const, above=of mu, xshift=0.5cm]     (taumu) {$10^{-3}$} ;
   \node[const, above=of tau, xshift=-0.5cm]   (atau)  {$10^{-3}$} ;
   \node[const, above=of tau, xshift=0.5cm]    (btau)  {$10^{-3}$} ;

   \factor[above=of y] {y-f} {left:$\mathcal{N}$} {mu,tau}     {y};
   \factor[above=of mu] {} {left:$\mathcal{N}$}   {mumu,taumu} {mu};
   \factor[above=of tau] {} {left:$\mathcal{G}$}  {atau,btau}  {tau};

   \plate {} {(y)(y-f)(y-f-caption)} {10} ;

                
.. code:: python

    from bayespy.nodes import GaussianARD
    GaussianARD(0, 1)



.. parsed-literal::

    <bayespy.inference.vmp.nodes.gaussian.GaussianARD at 0x7fa3343bce90>


