Mixture distribution
--------------------

.. math::

   \mathbf{x} 
   &\sim
   \mathrm{Mix}_{\mathcal{D}}
   \left(
       \lambda, 
       \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1}
   \right)

.. math::

   \lambda \in \{1, \ldots, N\},
   \quad \mathcal{D} \text{ is an exp.fam. distribution},
   \quad \mathbf{\Theta}^{(n)}_k \text{ are parameters of } \mathcal{D}

.. math::

   \log\mathrm{Mix}_{\mathcal{D}} 
   \left(
       \mathbf{x} 
       \left| \lambda, 
              \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1}
       \right.
   \right)
   &= 
   \sum^N_{n=1} [\lambda=n] 
   \mathbf{u}_{\mathcal{D}}(\mathbf{x})^{\mathrm{T}}
   \boldsymbol{\phi}_{\mathcal{D}}
   \left(
       \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K
   \right)
   \\
   & \quad +
   \sum^N_{n=1} [\lambda=n] 
   g_{\mathcal{D}} 
   \left(
       \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K
   \right)
   + f_{\mathcal{D}} (\mathbf{x})

.. math::

   \mathbf{u} (\mathbf{x})
   &=
   \mathbf{u}_{\mathcal{D}} (\mathbf{x})
   \\
   \boldsymbol{\phi} 
   \left(
       \lambda, 
       \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1}
   \right)
   &=
   \sum^N_{n=1} [\lambda=n] 
   \boldsymbol{\phi}_{\mathcal{D}}
   \left(
       \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K
   \right)
   %
   \\
   %
   \boldsymbol{\phi}_{\lambda}
   \left(
     \mathbf{x},
     \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1}
   \right)
   &=
   \left[\begin{matrix}
       \mathbf{u}_{\mathcal{D}} (\mathbf{x})^{\mathrm{T}}
       \boldsymbol{\phi}_{\mathcal{D}}
       \left(
         \mathbf{\Theta}^{(1)}_1, \ldots, \mathbf{\Theta}^{(1)}_K
       \right)
       + g_{\mathcal{D}}
       \left(
         \mathbf{\Theta}^{(1)}_1, \ldots, \mathbf{\Theta}^{(1)}_K
       \right)
       \\
       \vdots
       \\
       \mathbf{u}_{\mathcal{D}} (\mathbf{x})^{\mathrm{T}}
       \boldsymbol{\phi}_{\mathcal{D}}
       \left(
         \mathbf{\Theta}^{(N)}_1, \ldots, \mathbf{\Theta}^{(N)}_K
       \right)
       + g_{\mathcal{D}}
       \left(
         \mathbf{\Theta}^{(N)}_1, \ldots, \mathbf{\Theta}^{(N)}_K
       \right)
   \end{matrix}\right]
   %
   \\
   %
   \boldsymbol{\phi}_{\mathbf{\Theta}^{(m)}_l} 
   \left(
     \mathbf{x},
     \lambda, 
     \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1} 
     \setminus \left\{ \mathbf{\Theta}^{(m)}_l \right\}
   \right)
   &=
   [\lambda=m] \boldsymbol{\phi}_{\mathcal{D}\rightarrow\mathbf{\Theta}_l}
   \left(
     \mathbf{x},
     \left\{ \mathbf{\Theta}^{(m)}_k \right\}_{k\neq l}
   \right)
   %
   \\
   %
   g
   \left(
     \lambda, 
     \left\{ \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K \right\}^N_{n=1}
   \right)
   &=
   \sum^N_{n=1} [\lambda=n] 
   g_{\mathcal{D}} 
   \left(
     \mathbf{\Theta}^{(n)}_1, \ldots, \mathbf{\Theta}^{(n)}_K
   \right)
   \\
   g (\boldsymbol{\phi})
   &=
   g_{\mathcal{D}} (\boldsymbol{\phi})
   \\
   f(\mathbf{x})
   &=
   f_{\mathcal{D}} (\mathbf{x})
   \\
   \overline{\mathbf{u}}  (\boldsymbol{\phi})
   &=
   \overline{\mathbf{u}}_{\mathcal{D}}  (\boldsymbol{\phi})
   


