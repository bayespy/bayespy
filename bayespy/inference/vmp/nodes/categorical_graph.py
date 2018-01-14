################################################################################
# Copyright (C) 2017 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import junctiontree as jt


from .node import Moments
from .stochastic import Distribution
from bayespy.utils import misc


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

        # FIXME: Plates not supported yet
        if plates != {}:
            raise NotImplementedError("Plates not yet implemented")

        self._factors = [
            variable.get("given", []) + [name]
            for (name, variable) in dag.items()
        ]

        self._original_sizes = {
            name: np.shape(variable["table"])[-1]
            for (name, variable) in dag.items()
        }

        self._variable_factors = {
            variable: [
                (index, self._factors[index].index(variable) - len(self._factors[index]))
                for index in range(len(self._factors))
                if variable in self._factors[index]
            ]
            for variable in self._original_sizes.keys()
        }

        # State
        self._junctiontree = None
        self._sizes = self._original_sizes
        self._slice_potentials = lambda xs: xs
        self._unslice_potentials = lambda xs: xs
        self.u = None

        # FIXME: Here we just assume fixed arrays as CPTs, not Dirichlet nodes
        # supported yet.
        self._cpts = [
            variable["table"]
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

        # Create a function to slice the potential arrays
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
        self.u = None

        return


    def update(self):
        # TODO: Fetch CPTs from Dirichlet parents and make use of potentials
        # from children.

        if self._junctiontree is None:
            self._junctiontree = jt.create_junction_tree(
                self._factors,
                self._sizes
            )

        xs = self._slice_potentials(self._cpts)
        u = self._junctiontree.propagate(xs)
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
