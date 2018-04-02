################################################################################
# Copyright (C) 2017 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import junctiontree as jt
import attr


from .node import Node
from .stochastic import Distribution

from .dirichlet import DirichletMoments

from bayespy.utils import misc


@attr.s(frozen=True, slots=True)
class Variable():


    table = attr.ib(converter=lambda x: Node._ensure_moments(x, DirichletMoments))
    given = attr.ib(converter=tuple, default=())
    plates = attr.ib(converter=tuple, default=())

    # @plates.validator
    # def check(self, attribute, value):
    #     pass


class CategoricalGraph():
    """

    Some future ideas:

    X = CategoricalGraph(
        [
            Variable(
                table=[0.8, 0.2],
            ),
            Variable(
                table=[[0.6, 0.4], [0.99, 0.01]],
                given=[0],
                plates=['lawns']
            ),
            Variable(
                table=[ [[1.0, 0.0], [0.2, 0.8]],
                        [[0.1, 0.9], [0.01, 0.99]] ],
                given=[0, 1],
                plates=['lawns']
            ),
        ]
        {
            'rain': [0.8, 0.2],
            'sprinkler': dict(table=[[0.6, 0.4], [0.99, 0.01]],
                            given=['rain'],
                            plates=['lawns'],
                            platemap="i->i"),
            'grass wet': dict(table=[ [[1.0, 0.0], [0.2, 0.8]],
                                    [[0.1, 0.9], [0.01, 0.99]] ],
                            given=['sprinkler', 'rain'],
                            plates=['lawns'],
                            platemap="i->i")
        },
        plates={
            'lawns': 10,
        },
        # If needed, one could explicitly give a mapping from the graph plates
        # to plate axes of other BayesPy nodes:
        # NO! Use platemap attribute in the Variable class!
        plates_axes={
            'lawns': 0,
        },
    )

    Hmm.. How to get a child node with arbitrary (joint) marginals?

    X[['rain','sprinkler'],['rain']] ?

    """


    def __init__(self, dag, plates={}):

        # Convert to Variables
        dag = {
            name: Variable(**config)
            for (name, config) in dag.items()
        }

        # Validate plates (children must have those plates that the parents have)

        # Validate shapes of the CPTs

        # Validate that plate keys and variable keys are unique

        # Mapping: factor -> variables
        #
        # The name of the variables contained in each factor. Each CPT means a
        # factor which contains the variable itself and its parents.
        self._factors = [
            variable.plates + variable.given + (name,)
            for (name, variable) in dag.items()
        ]

        # Mapping: variable -> factors
        #
        # Reverse mapping, should be done in junctiontree package? For each
        # variable, find the list of factors in which the variable is included.
        self._variable_factors = {
            variable: [
                (index, self._factors[index].index(variable) - len(self._factors[index]))
                for index in range(len(self._factors))
                if variable in self._factors[index]
            ]
            for variable in dag.keys()
        }

        # Number of states for each variable (CPTs are assumed Dirichlet moments here)
        variable_sizes = {
            name: variable.table.dims[0][0]
            for (name, variable) in dag.items()
        }

        # Sizes of all axes (variables and plates), that is, just combine the
        # two size dicts
        all_sizes = list(variable_sizes.items()) + list(plates.items())
        self._original_sizes = {
            key: size for (key, size) in all_sizes
        }

        # State
        self._junctiontree = None
        self._sizes = self._original_sizes
        self._slice_potentials = lambda xs: xs
        self._unslice_potentials = lambda xs: xs
        self.u = {
            variable: np.nan for variable in self._variable_factors.keys()
        }

        # FIXME: Here we just assume fixed arrays as CPTs, not Dirichlet nodes
        # supported yet.
        self._cpts = [
            variable.table
            for variable in dag.values()
        ]

        # TODO: Call super?
        return


    def lower_bound_contribution(self):
        raise NotImplementedError()


    def get_moments(self):
        return self.u


    def observe(self, y):
        """Give dictionary like {"rain": 1, "sunny": 0}.

        NOTE: Previously set observed states are reset.
        """

        # Create a function to slice the potential arrays. This is used for
        # observing a variable: only use that state which was observed.
        def slice_potentials(xs):
            xs = xs.copy()
            for (variable, ind) in y.items():
                for (factor, axis) in self._variable_factors[variable]:
                    xs[factor] = np.take(xs[factor], [ind], axis=axis)
            return xs


        def unslice_potentials(xs):
            xs = xs.copy()
            for (variable, ind) in y.items():
                for (factor, axis) in self._variable_factors[variable]:
                    e = misc.eye(
                        index=ind,
                        size=self._original_sizes[variable],
                        axis=axis,
                    )
                    xs[factor] = e * xs[factor]
            return xs


        # Modify sizes
        self._sizes = self._original_sizes.copy()
        self._sizes.update({key: 1 for key in y.keys()})

        # Junction tree needs to be rebuilt
        self._junctiontree = None
        self._slice_potentials = slice_potentials
        self._unslice_potentials = unslice_potentials
        self.u = {
            variable: np.nan for variable in self._variable_factors.keys()
        }

        return


    def update(self):
        # TODO: Fetch CPTs from Dirichlet parents and make use of potentials
        # from children.

        # FIXME: Convert to lists.. Fix this in junctiontree
        factors = [list(f) for f in self._factors]

        if self._junctiontree is None:
            self._junctiontree = jt.create_junction_tree(
                #self._factors,
                factors,
                self._sizes
            )

        # Get the numerical probability tables from the Dirichlet nodes
        #
        # FIXME: Convert <log p> to exp( <log p> ). Perhaps junctiontree
        # package could support logarithms of the probabilities? Also, note
        # that these don't sum to one, they are non-normalized probabilities.
        cpts = [np.exp(cpt.get_moments()[0]) for cpt in self._cpts]

        xs = self._slice_potentials(cpts)
        # Convert to lists..
        u = self._junctiontree.propagate(list(xs))
        u = [ui / np.sum(ui) for ui in self._unslice_potentials(u)]

        # For simplicity - and temporarily - marginalize each potential for the variable
        self.u = {
            variable: misc.sum_product(
                u[self._variable_factors[variable][0][0]],
                axes_to_keep=[self._variable_factors[variable][0][1]]
            )
            for variable in self._sizes.keys()
        }

        return
