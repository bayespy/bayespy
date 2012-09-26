Wishart distribution
--------------------

.. math::

   \mathbf{\Lambda} 
   &\sim \mathcal{W}(n, \mathbf{V}),

.. math::
   n > D-1,
   \quad \mathbf{\Lambda}, \mathbf{V} \in \mathbb{R}^{D \times D},
   \quad \mathbf{\Lambda}, \mathbf{V} \text{ symmetric positive definite}

.. math::

   \log\mathcal{W}( \mathbf{\Lambda} | n, \mathbf{V} )
   &= 
   - \frac{1}{2} \operatorname{tr} (\mathbf{\Lambda V})
   + \frac{n}{2} \log |\mathbf{\Lambda}|
   + \frac{n}{2} \log |\mathbf{V}|
   - \frac{D+1}{2} \log |\mathbf{\Lambda}|
   - \frac{nD}{2} \log 2
   - \log \Gamma_D \left(\frac{n}{2}\right)

.. math::

   \mathbf{u} (\mathbf{\Lambda})
   &=
   \left[ \begin{matrix}
     \mathbf{\Lambda}
     \\
     \log |\mathbf{\Lambda}|
   \end{matrix} \right]
   \\
   \boldsymbol{\phi} (n, \mathbf{V})
   &=
   \left[ \begin{matrix}
     - \frac{1}{2} \mathbf{V}
     \\
     \frac{1}{2} n
   \end{matrix} \right]
   \\
   \boldsymbol{\phi}_{n} (\mathbf{\Lambda}, \mathbf{V})
   &=
   \left[ \begin{matrix}
     \frac{1}{2}\log|\mathbf{\Lambda}|
     + \frac{1}{2}\log|\mathbf{V}|
     + \frac{D}{2} \log 2
     \\
     -1
   \end{matrix} \right]
   \\
   \boldsymbol{\phi}_{\mathbf{V}} (\mathbf{\Lambda}, n)
   &=
   \left[ \begin{matrix}
     - \frac{1}{2} \mathbf{\Lambda}
     \\
     \frac{1}{2} n
   \end{matrix} \right]
   \\
   g (n, \mathbf{V})
   &=
   \frac{n}{2} \log|\mathbf{V}| 
   - \frac{nD}{2}\log 2 
   - \log \Gamma_D \left(\frac{n}{2}\right)
   \\
   g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
   &=
   \boldsymbol{\phi}_2 \log|-\boldsymbol{\phi}_1|
   - \log \Gamma_D (\boldsymbol{\phi}_2)
   \\
   f(\mathbf{\Lambda})
   &=
   - \frac{D+1}{2} \log|\mathbf{\Lambda}|
   \\
   \overline{\mathbf{u}}  (\boldsymbol{\phi})
   &=
   \left[ \begin{matrix}
     - \boldsymbol{\phi}_2 \boldsymbol{\phi}^{-1}_1
     \\
     - \log|-\boldsymbol{\phi}_1|
     + \psi_D(\boldsymbol{\phi}_2)
   \end{matrix} \right]
   

