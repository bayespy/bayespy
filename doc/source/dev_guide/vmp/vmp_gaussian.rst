Multivariate normal distribution
--------------------------------

.. math::

   \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),

.. math::

   \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, 
   \quad \mathbf{\Lambda} \in \mathbb{R}^{D \times D},
   \quad \mathbf{\Lambda} \text{ symmetric positive definite}

.. math::

   \log\mathcal{N}( \mathbf{x} | \boldsymbol{\mu}, \mathbf{\Lambda} )
   &= 
   - \frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{\Lambda} \mathbf{x}
   + \mathbf{x}^{\mathrm{T}} \mathbf{\Lambda} \boldsymbol{\mu}
   - \frac{1}{2} \boldsymbol{\mu}^{\mathrm{T}} \mathbf{\Lambda}
     \boldsymbol{\mu}
   + \frac{1}{2} \log |\mathbf{\Lambda}|
   - \frac{D}{2} \log (2\pi)

.. math::

   \mathbf{u} (\mathbf{x})
   &=
   \left[ \begin{matrix}
     \mathbf{x}
     \\
     \mathbf{xx}^{\mathrm{T}}
   \end{matrix} \right]
   \\
   \boldsymbol{\phi} (\boldsymbol{\mu}, \mathbf{\Lambda})
   &=
   \left[ \begin{matrix}
     \mathbf{\Lambda} \boldsymbol{\mu} 
     \\
     - \frac{1}{2} \mathbf{\Lambda}
   \end{matrix} \right]
   \\
   \boldsymbol{\phi}_{\boldsymbol{\mu}} (\mathbf{x}, \mathbf{\Lambda})
   &=
   \left[ \begin{matrix}
     \mathbf{\Lambda} \mathbf{x} 
     \\
     - \frac{1}{2} \mathbf{\Lambda}
   \end{matrix} \right]
   \\
   \boldsymbol{\phi}_{\mathbf{\Lambda}} (\mathbf{x}, \boldsymbol{\mu})
   &=
   \left[ \begin{matrix}
     - \frac{1}{2} \mathbf{xx}^{\mathrm{T}}
     + \frac{1}{2} \mathbf{x}\boldsymbol{\mu}^{\mathrm{T}}
     + \frac{1}{2} \boldsymbol{\mu}\mathbf{x}^{\mathrm{T}}
     - \frac{1}{2} \boldsymbol{\mu\mu}^{\mathrm{T}}
     \\
     \frac{1}{2}
   \end{matrix} \right]
   \\
   g (\boldsymbol{\mu}, \mathbf{\Lambda})
   &=
   - \frac{1}{2} \operatorname{tr}(\boldsymbol{\mu\mu}^{\mathrm{T}}
                                   \mathbf{\Lambda} )
   + \frac{1}{2} \log |\mathbf{\Lambda}|
   \\
   g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
   &=
   \frac{1}{4} \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
   \boldsymbol{\phi}_1
   + \frac{1}{2} \log | -2 \boldsymbol{\phi}_2 |
   \\
   f(\mathbf{x})
   &= - \frac{D}{2} \log(2\pi)
   \\
   \overline{\mathbf{u}}  (\boldsymbol{\phi})
   &=
   \left[ \begin{matrix}
     - \frac{1}{2} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
     \\
     \frac{1}{4} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
     \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
     - \frac{1}{2} \boldsymbol{\phi}^{-1}_2
   \end{matrix} \right]
   

