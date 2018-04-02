################################################################################
# Copyright (C) 2018 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `categorical_graph` module.
"""

import warnings

import numpy as np
import scipy

from bayespy.inference.vmp.nodes.dirichlet import DirichletMoments
from bayespy.nodes import CategoricalGraph, Dirichlet

from bayespy.utils import random
from bayespy.utils import misc

from bayespy.utils.misc import TestCase


def assertDictionaryMoments(X, y):
    x = X.get_moments()
    assert set(x.keys()) == set(y.keys())
    for key in x.keys():
        np.testing.assert_allclose(x[key], y[key])
    return


def normalize(p):
    return p / np.sum(p, axis=-1, keepdims=True)


def random(*shape):
    p = np.random.rand(*shape)
    return normalize(np.random.rand(*shape))


def sumproduct(*args):
    return normalize(np.einsum(*args))


def onehot(ind, size):
    e = np.zeros(size)
    e[ind] = 1
    return e


class TestCategorical(TestCase):
    """
    Unit tests for Categorical node
    """


    def test_moments(self):

        def _run(dag, u, ys, **kwargs):

            def _check(X, y):
                X.update()
                cpts = {
                    name:
                    np.exp(Dirichlet._ensure_moments(config["table"], DirichletMoments).get_moments()[0])
                    for (name, config) in dag.items()
                }
                for (name, ind) in y.items():
                    cpts[name] = cpts[name] * onehot(ind, cpts[name].shape[-1])
                assertDictionaryMoments(X, u(cpts))

            X = CategoricalGraph(dag, **kwargs)
            _check(X, {})
            for y in ys:
                X.observe(y)
                _check(X, y)

            return

        # FIXME: Empty graph not yet supported in junctiontree
        # _run({}, {})

        # Single variable
        #
        # x
        _run(
            {
                "x": {
                    "table": random(2),
                }
            },
            lambda cpts: {"x": normalize(cpts["x"])},
            [
                {"x": 1},
            ]
        )

        # Pair
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": random(2, 3),
                },
            },
            lambda cpts: {
                "x": sumproduct("x,xy->x", cpts["x"], cpts["y"]),
                "y": sumproduct("x,xy->y", cpts["x"], cpts["y"]),
            },
            [
                {"x": 1},
                {"y": 0},
                {"x": 1, "y": 2},
            ]
        )

        # Separate graphs
        #
        # x y
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "table": random(3),
                },
            },
            lambda cpts: {
                "x": normalize(cpts["x"]),
                "y": normalize(cpts["y"]),
            },
            [
                {"x": 1},
                {"y": 1},
                {"x": 1, "y": 2},
            ]
        )

        # Simple chain
        #
        # x
        # |
        # y
        # |
        # z
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": random(2, 3),
                },
                "z": {
                    "given": ["y"],
                    "table": random(3, 4),
                },
            },
            lambda cpts: {
                "x": sumproduct("x,xy,yz->x", cpts["x"], cpts["y"], cpts["z"]),
                "y": sumproduct("x,xy,yz->y", cpts["x"], cpts["y"], cpts["z"]),
                "z": sumproduct("x,xy,yz->z", cpts["x"], cpts["y"], cpts["z"]),
            },
            [
                {"x": 1},
                {"y": 2},
                {"z": 3},
                {"x": 1, "y": 2},
                {"x": 1, "z": 3},
                {"y": 2, "z": 3},
                {"x": 1, "y": 2, "z": 3},
            ],
        )

        # Diamond
        #
        #   x
        #  / \
        # y   z
        #  \ /
        #   v
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": random(2, 3),
                },
                "z": {
                    "given": ["x"],
                    "table": random(2, 4),
                },
                "v": {
                    "given": ["y", "z"],
                    "table": random(3, 4, 5),
                },
            },
            lambda cpts: {
                "x": sumproduct("x,xy,xz,yzv->x", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "y": sumproduct("x,xy,xz,yzv->y", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "z": sumproduct("x,xy,xz,yzv->z", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "v": sumproduct("x,xy,xz,yzv->v", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
            },
            [
                {"x": 1},
                {"y": 2},
                {"z": 3},
                {"v": 0},
                {"x": 1, "y": 2},
                {"x": 1, "z": 3},
                {"x": 1, "v": 0},
                {"y": 2, "z": 3},
                {"y": 2, "v": 0},
                {"z": 3, "v": 0},
                {"x": 1, "y": 2, "z": 3},
                {"x": 1, "y": 2, "v": 0},
                {"x": 1, "z": 3, "v": 0},
                {"y": 2, "z": 3, "v": 0},
                {"x": 1, "y": 2, "z": 3, "v": 0},
            ]
        )

        # Chain with shortcut (check that x gets propagated to v properly)
        #
        # x
        # |\
        # y \
        # | |
        # z /
        # |/
        # v
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": random(2, 3),
                },
                "z": {
                    "given": ["y"],
                    "table": random(3, 4),
                },
                "v": {
                    "given": ["x", "z"],
                    "table": random(2, 4, 5),
                }
            },
            lambda cpts: {
                "x": sumproduct("x,xy,yz,xzv->x", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "y": sumproduct("x,xy,yz,xzv->y", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "z": sumproduct("x,xy,yz,xzv->z", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "v": sumproduct("x,xy,yz,xzv->v", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
            },
            [
                {"x": 1},
                {"y": 2},
                {"z": 3},
                {"v": 0},
                {"x": 1, "y": 2},
                {"x": 1, "z": 3},
                {"x": 1, "v": 0},
                {"y": 2, "z": 3},
                {"y": 2, "v": 0},
                {"z": 3, "v": 0},
                {"x": 1, "y": 2, "z": 3},
                {"x": 1, "y": 2, "v": 0},
                {"x": 1, "z": 3, "v": 0},
                {"y": 2, "z": 3, "v": 0},
                {"x": 1, "y": 2, "z": 3, "v": 0},
            ]
        )

        # Dirichlet parent
        _run(
            {
                "x": {
                    "table": Dirichlet([3, 7]),
                }
            },
            lambda cpts: {
                "x": normalize(cpts["x"])
            },
            [
                {"x": 1},
            ]
        )

        #
        # PLATES
        #

        # Single variable
        #
        # x
        _run(
            {
                "x": {
                    "table": random(10, 2),
                    "plates": ["a"],
                }
            },
            lambda cpts: {"x": normalize(cpts["x"])},
            [
                {"x": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
            ],
            plates={"a": 10},
        )

        # CPT broadcasted to plates

        # Error: CPT value negative

        # Error: CPT not summing to one

        # Error: CPT axis incorrect

        # Error: CPT extra axis

        # Error: CPT axis singleton

        # Error: Given name not in variables

        # Error: Same variable given multiple times

        # Error: Observation axis incorrect

        # Error: Observation extra axis

        # Error: Observation axis singleton

        pass


    def test_message_to_parent(self):
        pass
