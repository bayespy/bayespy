################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Module for the categorical Markov chain node.
"""

import numpy as np

from .deterministic import Deterministic
from .expfamily import (ExponentialFamily,
                        ExponentialFamilyDistribution,
                        useconstructor)
from .node import (Moments,
                   ensureparents)
from .categorical import CategoricalMoments
from .dirichlet import (Dirichlet,
                        DirichletMoments)

from bayespy.utils import misc, random

class CategoricalMarkovChainMoments(Moments):
    """
    Class for the moments of categorical Markov chain variables.
    """


    def __init__(self, categories, length):
        """
        Create moments object for categorical Markov chain variables.
        """
        self.categories = categories
        self.length = length
        self.dims = ( (categories,), (length-1, categories, categories) )
        return


    def compute_fixed_moments(self, x):
        """
        Compute the moments for a fixed value
        """
        raise NotImplementedError("compute_fixed_moments not implemented for "
                                  "%s" 
                                  % (self.__class__.__name__))


    @classmethod
    def from_values(cls, x, categories):
        """
        Return the shape of the moments for a fixed value.
        """
        raise NotImplementedError("from_values not implemented "
                                  "for %s"
                                  % (self.__class__.__name__))


class CategoricalMarkovChainDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of categorical Markov chain variables.
    """    


    def __init__(self, categories, states):
        """
        Create VMP formula node for a categorical variable

        `categories` is the total number of categories.
        `states` is the length of the chain.
        """
        self.K = categories
        self.N = states

    def compute_message_to_parent(self, parent, index, u, u_p0, u_P):
        """
        Compute the message to a parent node.
        """
        if index == 0:
            return [ u[0] ]
        elif index == 1:
            return [ u[1] ]
        else:
            raise ValueError("Parent index out of bounds")

    def compute_weights_to_parent(self, index, weights):
        """
        Maps the mask to the plates of a parent.
        """
        if index == 0:
            return weights
        elif index == 1:
            # Add plate axis for the time axis and row axis of the transition
            # matrix
            return np.asanyarray(weights)[...,None,None]
        else:
            raise ValueError("Parent index out of bounds")

    def compute_phi_from_parents(self, u_p0, u_P, mask=True):
        """
        Compute the natural parameter vector given parent moments.
        """
        phi0 = u_p0[0]
        phi1 = u_P[0] * np.ones((self.N-1,self.K,self.K))
        return [phi0, phi1]

    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        logp0 = phi[0]
        logP = phi[1]
        (z0, zz, cgf) = random.alpha_beta_recursion(logp0, logP)
        u = [z0, zz]
        return (u, cgf)

    def compute_cgf_from_parents(self, u_p0, u_P):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        return 0
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        raise NotImplementedError()

    def plates_to_parent(self, index, plates):
        """
        Resolves the plate mapping to a parent.

        Given the plates of the node's moments, this method returns the plates
        that the message to a parent has for the parent's distribution.
        """
        if index == 0:
            return plates
        elif index == 1:
            return plates + (self.N-1, self.K)
        else:
            raise ValueError("Parent index out of bounds")
        
    def plates_from_parent(self, index, plates):
        """
        Resolve the plate mapping from a parent.
        
        Given the plates of a parent's moments, this method returns the plates
        that the moments has for this distribution.
        """
        if index == 0:
            return plates
        elif index == 1:
            return plates[:-2]
        else:
            raise ValueError("Parent index out of bounds")

        
    def random(self, *phi, plates=None):
        """
        Draw a random sample from the distribution.
        """
        # Convert natural parameters to transition probabilities
        p0 = np.exp(phi[0] - misc.logsumexp(phi[0], 
                                            axis=-1,
                                            keepdims=True))
        P = np.exp(phi[1] - misc.logsumexp(phi[1],
                                           axis=-1,
                                           keepdims=True))
        # Explicit broadcasting
        P = P * np.ones(plates)[...,None,None,None]
        # Allocate memory
        Z = np.zeros(plates + (self.N,), dtype=np.int)
        # Draw initial state
        Z[...,0] = random.categorical(p0, size=plates)
        # Create [0,1,2,...,len(plate_axis)] indices for each plate axis and
        # make them broadcast properly
        nplates = len(plates)
        plates_ind = [np.arange(plate)[(Ellipsis,)+(nplates-i-1)*(None,)]
                      for (i, plate) in enumerate(plates)]
        plates_ind = tuple(plates_ind)
        # Draw next states iteratively
        for n in range(self.N-1):
            # Select the transition probabilities for the current state but take
            # into account the plates.  This leads to complex NumPy
            # indexing.. :)
            time_ind = min(n, np.shape(P)[-3]-1)
            ind = plates_ind + (time_ind, Z[...,n], Ellipsis)
            p = P[ind]
            # Draw next state
            z = random.categorical(P[ind])
            Z[...,n+1] = z
            
        return Z

    
class CategoricalMarkovChain(ExponentialFamily):
    r"""
    Node for categorical Markov chain random variables.
    
    The node models a Markov chain which has a discrete set of K possible states
    and the next state depends only on the previous state and the state
    transition probabilities.  The graphical model is shown below:

    .. bayesnet::

       \tikzstyle{latent} += [minimum size=30pt];
       
       \node[latent] (x0) {$x_0$};
       \node[latent, right=of x0] (x1) {$x_1$};
       \node[right=of x1] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$x_{N-1}$};
       \edge {x0}{x1};
       \edge {x1}{dots};
       \edge {dots}{xn};

       \node[latent, above=of x0] (pi) {$\boldsymbol{\pi}$};
       \node[latent, above=of dots] (A) {$\mathbf{A}$};
       \edge {pi} {x0};
       \edge {A} {x1,dots,xn};

    where :math:`\boldsymbol{\pi}` contains the probabilities for the initial
    state and :math:`\mathbf{A}` is the state transition probability matrix.  It
    is possible to have :math:`\mathbf{A}` varying in time.

    .. math::

        p(x_0, \ldots, x_{N-1}) &= p(x_0) \prod^{N-1}_{n=1} p(x_n|x_{n-1}),

    where
    
    .. math::

        p(x_0=k) &= \pi_k, \quad \text{for } k \in \{0,\ldots,K-1\},
        \\
        p(x_n=j|x_{n-1}=i) &= a_{ij}^{(n-1)} \quad \text{for } n=1,\ldots,N-1,\,
        i\in\{1,\ldots,K-1\},\, j\in\{1,\ldots,K-1\}
        \\
        a_{ij}^{(n)} &= [\mathbf{A}_n]_{ij}

    This node can be used to construct hidden Markov models by using
    :class:`Mixture` for the emission distribution.

    Parameters
    ----------
    
    pi : Dirichlet-like node or (...,K)-array
    
        :math:`\boldsymbol{\pi}`, probabilities for the first
        state. :math:`K`-dimensional Dirichlet.
        
    A : Dirichlet-like node or (K,K)-array or (...,1,K,K)-array or (...,N-1,K,K)-array
    
        :math:`\mathbf{A}`, probabilities for state
        transitions. :math:`K`-dimensional Dirichlet with plates (K,) or
        (...,1,K) or (...,N-1,K).
        
    states : int, optional
    
        :math:`N`, the length of the chain.

    See also
    --------
    
    Categorical, Dirichlet, GaussianMarkovChain, Mixture,
    SwitchingGaussianMarkovChain
    """


    def __init__(self, pi, A, states=None, **kwargs):
        """
        Create categorical Markov chain
        """
        super().__init__(pi, A, states=states, **kwargs)


    @classmethod
    def _constructor(cls, p0, P, states=None, **kwargs):
        """
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        Becase the distribution and moments object depend on the number of
        categories, that is, they depend on the parent node, this method can be
        used to construct those objects.
        """

        p0 = cls._ensure_moments(p0, DirichletMoments)
        P = cls._ensure_moments(P, DirichletMoments)

        # Number of categories
        D = p0.dims[0][0]

        parent_moments = (p0._moments, P._moments)

        # Number of states
        if len(P.plates) < 2:
            if states is None:
                raise ValueError("Could not infer the length of the Markov "
                                 "chain")
            N = int(states)
        else:
            if P.plates[-2] == 1:
                if states is None:
                    N = 2
                else:
                    N = int(states)
            else:
                if states is not None and P.plates[-2]+1 != states:
                    raise ValueError("Given length of the Markov chain is "
                                     "inconsistent with the transition "
                                     "probability matrix")
                N = P.plates[-2] + 1

        if p0.dims != P.dims:
            raise ValueError("Initial state probability vector and state "
                             "transition probability matrix have different "
                             "size")

        if len(P.plates) < 1 or P.plates[-1] != D:
            raise ValueError("Transition probability matrix is not square")

        dims = ( (D,), (N-1,D,D) )

        parents = [p0, P]
        distribution = CategoricalMarkovChainDistribution(D, N)
        moments = CategoricalMarkovChainMoments(D, N)

        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, p0.plates),
                                  distribution.plates_from_parent(1, P.plates)),
                distribution,
                moments,
                parent_moments)


class CategoricalMarkovChainToCategorical(Deterministic):
    """
    A node for converting categorical MC moments to categorical moments.
    """


    def __init__(self, Z, **kwargs):
        """
        Create a categorical MC moments to categorical moments conversion node.
        """
        # Convert parent to proper type. Z must be a node.
        Z = self._ensure_moments(Z, CategoricalMarkovChainMoments)
        K = Z.dims[0][-1]
        dims = ( (K,), )
        self._moments = CategoricalMoments(K)
        self._parent_moments = (Z._moments,)
        super().__init__(Z, dims=dims, **kwargs)


    def _compute_moments(self, u_Z):
        """
        Compute the moments given the moments of the parents.
        """
        # Add time axis to p0
        p0 = u_Z[0][...,None,:]
        # Sum joint probability arrays to marginal probability vectors
        zz = u_Z[1]
        p = np.sum(zz, axis=-2)

        # Broadcast p0 and p to same shape, except the time axis
        plates_p0 = np.shape(p0)[:-2]
        plates_p = np.shape(p)[:-2]
        shape = misc.broadcasted_shape(plates_p0, plates_p) + (1,1)
        p0 = p0 * np.ones(shape)
        p = p * np.ones(shape)

        # Concatenate
        P = np.concatenate((p0,p), axis=-2)

        return [P]

    def _compute_message_to_parent(self, index, m, u_Z):
        """
        Compute the message to a parent.
        """
        m0 = m[0][...,0,:]
        m1 = m[0][...,1:,None,:]
        return [m0, m1]


    def _compute_weights_to_parent(self, index, weights):
        """
        Compute the mask used for messages sent to a parent.
        """
        if index == 0:
            # "Sum" over the last axis
            # TODO/FIXME: Check this. BUG I THINK.
            return np.sum(weights, axis=-1)
        else:
            raise ValueError("Parent index out of bounds")


    def _plates_to_parent(self, index):
        if index == 0:
            return self.plates[:-1]
        else:
            raise ValueError("Parent index out of bounds")
    
    def _plates_from_parent(self, index):
        if index == 0:
            N = self.parents[0].dims[1][0]
            return self.parents[0].plates + (N+1,)
        else:
            raise ValueError("Parent index out of bounds")


# Make use of the conversion node
CategoricalMarkovChainMoments.add_converter(CategoricalMoments,
                                            CategoricalMarkovChainToCategorical)
