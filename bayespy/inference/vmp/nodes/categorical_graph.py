################################################################################
# Copyright (C) 2017 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import junctiontree as jt
import attr


from .node import Node, Moments
from .stochastic import Distribution
from .deterministic import Deterministic

from .dirichlet import DirichletMoments
from .categorical import CategoricalMoments

from bayespy.utils import misc


class CategoricalGraphMoments(Moments):
    pass


@attr.s(frozen=True, slots=True)
class ConditionalProbabilityTable():


    variable = attr.ib()
    table = attr.ib(converter=lambda x: Node._ensure_moments(x, DirichletMoments))
    given = attr.ib(converter=tuple, default=())


@attr.s(frozen=True, slots=True)
class Factor():


    name = attr.ib()
    potential = attr.ib()
    variables = attr.ib(converter=tuple)


def find_index(xs, x):
    return xs.index(x) - len(xs)


def onehot(index, size, axis=-1, extradims=0):
    if extradims < 0:
        raise ValueError("extradims must be non-negative")
    index = np.reshape(index, np.shape(index) + (extradims + 1) * (1,))
    return 1.0 * np.moveaxis(
         np.arange(size) == index,
        -1,
        axis
    )


def map_to_shape(sizes, keys):
    return tuple(sizes[key] for key in keys)


class CategoricalGraph(Node):
    """DAG for categorical variables with exact inference

    The exact inference is uses the Junction tree algorithm.

    A simple example showing basically all available features:

    >>> dag = CategoricalGraph(
    ...     {
    ...         "x": {
    ...             "table": [0.4, 0.6],
    ...         },
    ...         "y": {
    ...             "given": ["x"],
    ...             "table": [ [0.1, 0.3, 0.6], [0.8, 0.1, 0.1] ],
    ...         },
    ...     },
    ...     marginals={
    ...         "marg_y": {
    ...             "variables": ["y"],
    ...         },
    ...     },
    ... )
    >>> dag.update()
    >>> print(dag["x"].p)
    >>> print(dag["y"].p)
    >>> print(dag["marg_y"].p)
    >>> dag.observe({"y": [1, 2, 0, 0, 2, 2, 1, 2, 1, 0]})
    >>> dag.update()
    >>> print(dag["x"].p)
    >>> print(dag["y"].p)
    >>> print(dag["marg_y"].p)

    """

    # TODO:
    #
    # - random
    # - message to parent
    # - explicit extra marginals: CategoricalGraph({...}, marginals={...})
    # - message to children
    # - multi-dimensional categorical moments, support in mixture
    # - compare performance to categoricalchain
    # - implement categorical tree as an example #15 and #20
    # - implement stochastic block model #51
    # - example graph #23


    def __init__(self, dag, marginals={}):

        self._id = Node._id_counter
        Node._id_counter += 1

        self._moments = CategoricalGraphMoments()

        # Convert to CPTs
        cpts = [
            ConditionalProbabilityTable(variable=name, **config)
            for (name, config) in dag.items()
        ]

        # Inform parents about this new child node
        for cpt in cpts:
            cpt.table._add_child(self, cpt.variable)

        # Validate shapes of the CPTs

        # Validate that plate keys and variable keys are unique

        # Add a factor for each CPT and requested marginal
        def get_potential_function(node):
            return lambda: np.exp(node.get_moments()[0])

        # All factors: CPTs and explicitly requested (joint) marginals
        self._factors = [
            Factor(
                name=cpt.variable,
                variables=cpt.given + (cpt.variable,),
                potential=get_potential_function(cpt.table)
            )
            for cpt in cpts
        ] + [
            Factor(name=name, potential=lambda: 1.0, **config)
            for (name, config) in marginals.items()
        ]

        self._factor_by_name = {
            factor.name: factor
            for factor in self._factors
        }

        # Mapping: factor -> keys in the factor
        #
        # This is required by Junctiontree package.
        self._keys_in_factor = [
            factor.variables
            for factor in self._factors
        ]

        # Mapping: variable -> factors
        #
        # FIXME: Reverse mapping, should be done in junctiontree package? For
        # each variable, find the list of factors in which the variable is
        # included.
        self._factors_with_variable = {
            cpt.variable: [
                (
                    # Factor ID
                    index,
                    # The axis of this variable in the CPT array (as a negative axis)
                    find_index(factor.variables, cpt.variable)
                )
                for (index, factor) in enumerate(self._factors)
                if cpt.variable in factor.variables
            ]
            for cpt in cpts
        }

        # Number of states for each variable (CPTs are assumed Dirichlet moments here)
        variable_sizes = {
            cpt.variable: cpt.table.dims[0][0]
            for cpt in cpts
        }
        self._parent_shapes = {
            cpt.variable: cpt.table.plates + cpt.table.dims[0]
            for cpt in cpts
        }

        # Sizes of all axes (variables and plates), that is, just combine the
        # two size dicts
        self._original_sizes = {
            key: size for (key, size) in variable_sizes.items()
        }
        self._factor_shapes = [
            map_to_shape(self._original_sizes, factor.variables)
            for factor in self._factors
        ]

        # State
        self._junctiontree = None
        self._sizes = self._original_sizes
        self._slice_potentials = lambda xs: xs
        self._unslice_potentials = lambda xs: xs
        self.u = {
            factor.name: np.nan for factor in self._factors
        }

        self._parent_moments = []

        return super().__init__()


    def _message_to_parent(self, variable, u_parent):
        shape = self._parent_shapes[variable]
        m0 = misc.sum_to_shape(self.u[variable], shape)
        return [m0]


    def lower_bound_contribution(self):
        raise NotImplementedError()


    def get_moments(self):
        return self.u


    def _message_to_child(self):
        return self.u


    def observe(self, y):
        """Give dictionary like {"rain": 1, "sunny": 0}.

        NOTE: Previously set observed states are reset.
        """

        # Create a function to slice the potential arrays. This is used for
        # observing a variable: only use that state which was observed.
        def slice_potentials(xs):
            xs = xs.copy()
            # Loop observations
            for (variable, ind) in y.items():
                ind = np.asarray(ind, dtype=np.int)
                # Loop all factors that contain the observed variable
                for (factor, axis) in self._factors_with_variable[variable]:
                    xs[factor] = np.take(
                        xs[factor],
                        [ind],
                        axis=axis
                    )
            return xs


        def unslice_potentials(xs):
            xs = xs.copy()
            for (variable, ind) in y.items():
                for (factor, axis) in self._factors_with_variable[variable]:
                    e = onehot(
                        index=ind,
                        size=self._original_sizes[variable],
                        extradims=np.ndim(xs[factor]) - 1,
                        axis=axis,
                    )
                    xs[factor] = e * xs[factor]
            return xs

        # TODO: Validate observation array shapes

        # Modify sizes
        self._sizes = self._original_sizes.copy()
        self._sizes.update({key: 1 for key in y.keys()})

        # Junction tree needs to be rebuilt
        self._junctiontree = None
        self._slice_potentials = slice_potentials
        self._unslice_potentials = unslice_potentials
        self.u = {
            factor.name: np.nan for factor in self._factors
        }

        return


    def update(self):
        # TODO: Fetch CPTs from Dirichlet parents and make use of potentials
        # from children.

        # FIXME: Convert to lists.. Fix this in junctiontree
        factors = [list(f) for f in self._keys_in_factor]

        if self._junctiontree is None:
            self._junctiontree = jt.create_junction_tree(
                #self._keys_in_factor,
                factors,
                self._sizes
            )

        # Get the numerical probability tables from the Dirichlet nodes
        #
        # FIXME: Convert <log p> to exp( <log p> ). Perhaps junctiontree
        # package could support logarithms of the probabilities? Also, note
        # that these don't sum to one, they are non-normalized probabilities.
        potentials = [
            np.broadcast_to(factor.potential(), shape)
            for (shape, factor) in zip(self._factor_shapes, self._factors)
        ]

        xs = self._slice_potentials(potentials)
        # Convert to lists..
        u = self._junctiontree.propagate(list(xs))

        def _normalize(p):
            return p / np.sum(p, keepdims=True)

        self.u = {
            factor.name: _normalize(ui)
            for (factor, ui) in zip(self._factors, self._unslice_potentials(u))
        }

        return


    def __getitem__(self, name):
        return CategoricalMarginal(graph=self, name=name)


    def _get_id_list(self):
        return [self._id]


class CategoricalMarginal(Deterministic):


    def __init__(self, graph, name, **kwargs):
        self.factor_name = name
        # TODO/FIXME: Fix support for multiaxes categoricals (joint probability tables)
        shape = map_to_shape(graph._original_sizes, graph._factor_by_name[name].variables)
        self._moments = CategoricalMoments(shape)
        self._parent_moments = [CategoricalGraphMoments()]
        return super().__init__(graph, **kwargs)


    def get_moments(self):
        return [self.parents[0].get_moments()[self.factor_name]]


    def _plates_from_parent(self, index):
        return ()


    def _message_from_parents(self):
        return self.parents[0]._message_to_child()
