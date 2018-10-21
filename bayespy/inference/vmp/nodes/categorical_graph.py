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
    logpotential = attr.ib()
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


def take(x, ind, axis):
    """
    Take elements along the last axis but apply the shape to the leading axes.
    That is, for ind with ndim=2:
    y[i,j] = x_ij[i,j,...,ind[i,j]] for i=1,...,M and j=1,...,N
    """
    x = np.asanyarray(x)
    ndim = np.ndim(x)
    if axis >= 0:
        axis = axis - ndim
    if axis >= 0 or axis < -ndim:
        raise ValueError("Axis out of bounds")
    shape = np.shape(x)
    plates = shape[:np.ndim(ind)]
    n_plates = len(plates)
    ind_plates = np.ix_(*[range(plate) for plate in plates])
    inds = list(ind_plates) + [...] + [np.asarray(ind)] + (abs(axis) - 1) * [slice(None)]
    return np.expand_dims(x[tuple(inds)], axis)


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
    ...         "marg_y": ["y"],
    ...     },
    ...     plates=(10,)
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


    def __init__(self, dag, marginals={}, initialize=True, plates=(), **kwargs):

        self._id = Node._id_counter
        Node._id_counter += 1

        self._moments = CategoricalGraphMoments()

        # Create keys for plates. NOTE: These should be different from the keys
        # used in dag..
        self._keys_in_plates = tuple(
            "_plate{0}".format(ind) for ind in range(len(plates))
        )

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
        def get_logpotential_function(node):
            return lambda: node.get_moments()[0]

        # All factors: CPTs and explicitly requested (joint) marginals
        self._factors = [
            Factor(
                name=cpt.variable,
                variables=cpt.given + (cpt.variable,),
                logpotential=get_logpotential_function(cpt.table)
            )
            for cpt in cpts
        ] + [
            Factor(name=name, logpotential=lambda: 0.0, variables=variables)
            for (name, variables) in marginals.items()
        ]

        self._marginals = marginals.keys()

        self._factor_by_name = {
            factor.name: factor
            for factor in self._factors
        }

        self._factor_id_by_name = {
            factor.name: ind
            for (ind, factor) in enumerate(self._factors)
        }

        # Mapping: factor -> keys in the factor
        #
        # This is required by Junctiontree package.
        self._keys_in_factor = [
            self._keys_in_plates + factor.variables
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
        self._original_sizes.update(
            {
                plate_key: plate_size
                for (plate_key, plate_size) in zip(self._keys_in_plates, plates)
            }
        )
        self._factor_shapes = [
            plates + map_to_shape(self._original_sizes, factor.variables)
            for factor in self._factors
        ]

        # State
        self._junctiontree = None
        self._sizes = self._original_sizes
        self._slice_potentials = lambda xs: xs
        self._unslice_potentials = lambda xs: xs
        self.u = {
            factor.name: np.full(shape, np.nan)
            for (shape, factor) in zip(self._factor_shapes, self._factors)
        }

        self._parent_moments = []

        super().__init__(plates=plates, **kwargs)

        if initialize:
            self.update()

        return


    def _message_to_parent(self, variable, u_parent):
        shape = self._parent_shapes[variable]
        m0 = misc.sum_to_shape(self.u[variable], shape)
        return [m0]


    def lower_bound_contribution(self):
        logp = [factor.logpotential() for factor in self._factors]
        # FIXME: Probably not correct.
        return sum(
            (
                np.inner(
                    np.ravel(ui),
                    np.ravel(
                        # <log p> (prior term)
                        self._factor_by_name[fname].logpotential()
                        # -<log q> (entropy term) (also, avoid log(0))
                        - np.log(np.where(ui == 0, 1.0, ui))
                    )

                )
                for (fname, ui) in self.u.items()
            )
        )


    def get_moments(self):
        return {
            name: (
                # If explicit marginal, return as it is
                p if name in self._marginals else
                # If it is based on a CPT, marginalize so that only the
                # variable itself is left. But keep the plates!
                np.sum(p, axis=tuple(range(len(self.plates), np.ndim(p)-1)))
            )
            for (name, p) in self.u.items()
        }


    def _message_to_child(self):
        return self.get_moments()


    def _message_from_children(self):

        # Collect messages from children
        ms = [
            child._message_to_parent(index)
            for (child, index) in self.children
        ]

        # Convert and fold message dictionaries to a list
        msg = [0] * len(self._factors)
        for m in ms:
            for (key, value) in m.items():
                ind = self._factor_id_by_name[key]
                if key in self._marginals:
                    msg[ind] += value
                else:
                    factor = self._factors[ind]
                    newaxes = (len(factor.variables) - 1) * (np.newaxis,)
                    msg[ind] += value[(Ellipsis,) + newaxes + (slice(None),)]
        return msg



    def observe(self, y):
        """Give dictionary like {"rain": 1, "sunny": 0}.

        NOTE: Previously set observed states are reset.
        """

        self.observed = len(y.keys()) > 0

        # Create a function to slice the potential arrays. This is used for
        # observing a variable: only use that state which was observed.
        def slice_potentials(xs):
            xs = xs.copy()
            # Loop observations
            for (variable, ind) in y.items():
                ind = np.asarray(ind, dtype=np.int)
                # Loop all factors that contain the observed variable
                for (factor, axis) in self._factors_with_variable[variable]:
                    xs[factor] = take(
                        xs[factor],
                        ind,
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
                        extradims=np.ndim(xs[factor]) - len(self.plates) - 1,
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

        if self._junctiontree is None:
            self._junctiontree = jt.create_junction_tree(
                # FIXME: Convert to lists.. Fix this in junctiontree
                [list(f) for f in self._keys_in_factor],
                self._sizes
            )

        # Get the messages from children
        child_msg = self._message_from_children()

        def exp(x):
            # Numerically stable way to calculate exp. Note that scaling of the
            # result is changed. Plates are handled independently.
            axs = tuple(range(len(self.plates), np.ndim(x)))
            return np.exp(x - np.amax(x, axis=axs, keepdims=True))

        # Get the numerical probability tables from the Dirichlet nodes
        #
        # FIXME: Convert <log p> to exp( <log p> ). Perhaps junctiontree
        # package could support logarithms of the probabilities? Also, note
        # that these don't sum to one, they are non-normalized probabilities.
        potentials = [
            exp(
                np.broadcast_to(
                    factor.logpotential() + child_m,
                    shape
                )
            )
            for (shape, factor, child_m) in zip(self._factor_shapes, self._factors, child_msg)
        ]

        xs = self._slice_potentials(potentials)
        # Convert to lists..
        u = self._junctiontree.propagate(list(xs))

        def _normalize(p):
            norm_axes = tuple(range(len(self.plates), np.ndim(p)))
            return p / np.sum(p, keepdims=True, axis=norm_axes)

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
        #
        categories = (
            # If explicit marginal, return as it is
            shape if name in graph._marginals else
            # If it is based on a CPT, marginalize so that only the variable
            # itself is left
            shape[-1:]
        )
        self._moments = CategoricalMoments(categories)
        self._parent_moments = [CategoricalGraphMoments()]
        return super().__init__(graph, dims=self._moments.dims, **kwargs)


    def get_moments(self):
        return [self.parents[0].get_moments()[self.factor_name]]


    def _message_to_parent(self, index, u_parent=None):
        return {self.factor_name: self._message_from_children()[0]}


    def _message_to_child(self):
        return self.get_moments()


    def _plates_from_parent(self, index):
        return self.parents[0].plates


    def _message_from_parents(self):
        return self.parents[0]._message_to_child()
