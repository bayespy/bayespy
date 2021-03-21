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
from bayespy.nodes import CategoricalGraph, Dirichlet, Mixture, GaussianARD, Bernoulli
from bayespy.inference import VB

from ..categorical_graph import onehot

from bayespy.utils import random as bp_random
from bayespy.utils import misc

from bayespy.utils.misc import TestCase


def assertDictionaryMoments(X, y):
    x = X.get_moments()
    assert set(x.keys()) == set(y.keys())
    for key in x.keys():
        np.testing.assert_allclose(x[key], y[key])
    return


def normalize(p, axis=None):
    return p / np.sum(p, axis=axis, keepdims=True)


def random(*shape):
    p = np.random.rand(*shape)
    return normalize(np.random.rand(*shape), axis=-1)


def sumproduct(*args, plates_ndim=0):
    y = np.einsum(*args)
    norm_ndim = np.ndim(y) - plates_ndim
    return normalize(y, axis=tuple(range(-norm_ndim, 0)))


class TestCategorical(TestCase):
    """
    Unit tests for Categorical node
    """


    def test_onehot(self):

        self.assertAllClose(
            onehot(2, 4),
            [0, 0, 1, 0]
        )
        self.assertAllClose(
            onehot(2, 4, extradims=2),
            [[[0, 0, 1, 0]]]
        )
        self.assertAllClose(
            onehot(2, 4, extradims=2, axis=-2),
            [[[0], [0], [1], [0]]]
        )
        self.assertAllClose(
            onehot([2, 1, 0], 4),
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
        )
        self.assertAllClose(
            onehot([2, 1, 0], 4, extradims=2),
            [
                [[[0, 0, 1, 0]]],
                [[[0, 1, 0, 0]]],
                [[[1, 0, 0, 0]]],
            ],
        )
        self.assertAllClose(
            onehot([2, 1, 0], 4, extradims=2, axis=0),
            [
                [ [[0]], [[0]], [[1]] ],
                [ [[0]], [[1]], [[0]] ],
                [ [[1]], [[0]], [[0]] ],
                [ [[0]], [[0]], [[0]] ],
            ]
        )
        self.assertAllClose(
            onehot([[2, 1, 0], [3, 3, 1]], 4),
            [
                [
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                ],
            ],
        )
        self.assertAllClose(
            onehot([[2, 1, 0], [3, 3, 1]], 4, extradims=2),
            [
                [
                    [[[0, 0, 1, 0]]],
                    [[[0, 1, 0, 0]]],
                    [[[1, 0, 0, 0]]],
                ],
                [
                    [[[0, 0, 0, 1]]],
                    [[[0, 0, 0, 1]]],
                    [[[0, 1, 0, 0]]],
                ],
            ],
        )
        return


    def test_moments(self):

        def _run(dag, u, ys, **kwargs):


            def to_cpt(X):
                return np.exp(
                    Dirichlet._ensure_moments(
                        X,
                        DirichletMoments
                    ).get_moments()[0]
                )


            def _check(X, y):
                X.update()
                cpts = {
                    name: to_cpt(config["table"])
                    for (name, config) in dag.items()
                }
                for (name, ind) in y.items():
                    cpts[name] = cpts[name] * onehot(
                        ind,
                        cpts[name].shape[-1],
                        extradims=len(dag[name].get("given", []))
                    )
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
                "name": {
                    "table": random(2),
                }
            },
            lambda cpts: {"name": normalize(cpts["name"])},
            [
                {"name": 1},
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

        # With custom marginals
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
                "marg_xyz": sumproduct("x,xy,yz->xyz", cpts["x"], cpts["y"], cpts["z"]),
                "marg_xy": sumproduct("x,xy,yz->xy", cpts["x"], cpts["y"], cpts["z"]),
                "marg_yz": sumproduct("x,xy,yz->yz", cpts["x"], cpts["y"], cpts["z"]),
                "marg_xz": sumproduct("x,xy,yz->xz", cpts["x"], cpts["y"], cpts["z"]),
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
            marginals={
                "marg_xyz": ["x", "y", "z"],
                "marg_xy": ["x", "y"],
                "marg_yz": ["y", "z"],
                "marg_xz": ["x", "z"],
            }
        )

        # Diamond with plates (some probability tables have the plates)
        #
        #   x
        #  / \
        # y   z
        #  \ /
        #   v
        x = [1, 1, 0, 1, 0, 1]
        y = [2, 0, 1, 2, 0, 2]
        z = [0, 3, 3, 3, 1, 2]
        v = [4, 4, 4, 4, 1, 0]
        _run(
            {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": random(6, 2, 3),
                },
                "z": {
                    "given": ["x"],
                    "table": random(6, 2, 4),
                },
                "v": {
                    "given": ["y", "z"],
                    "table": random(3, 4, 5),
                },
            },
            lambda _cpts: (
                (
                    lambda cpts: {
                        "x": sumproduct("ax,axy,axz,ayzv->ax", cpts["x"], cpts["y"], cpts["z"], cpts["v"], plates_ndim=1),
                        "y": sumproduct("ax,axy,axz,ayzv->ay", cpts["x"], cpts["y"], cpts["z"], cpts["v"], plates_ndim=1),
                        "z": sumproduct("ax,axy,axz,ayzv->az", cpts["x"], cpts["y"], cpts["z"], cpts["v"], plates_ndim=1),
                        "v": sumproduct("ax,axy,axz,ayzv->av", cpts["x"], cpts["y"], cpts["z"], cpts["v"], plates_ndim=1),
                    }
                )(
                    {
                        "x": np.broadcast_to(_cpts["x"], (6, 2)),
                        "y": np.broadcast_to(_cpts["y"], (6, 2, 3)),
                        "z": np.broadcast_to(_cpts["z"], (6, 2, 4)),
                        "v": np.broadcast_to(_cpts["v"], (6, 3, 4, 5)),
                    }
                )
            ),
            [
                {"x": x},
                {"y": y},
                {"z": z},
                {"v": v},
                {"x": x, "y": y},
                {"x": x, "z": z},
                {"x": x, "v": v},
                {"y": y, "z": z},
                {"y": y, "v": v},
                {"z": z, "v": v},
                {"x": x, "y": y, "z": z},
                {"x": x, "y": y, "v": v},
                {"x": x, "z": z, "v": v},
                {"y": y, "z": z, "v": v},
                {"x": x, "y": y, "z": z, "v": v},
            ],
            plates=(6,)
        )

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

        def _run(parents, dag, messages, observations, **kwargs):

            # Construct the DAG
            dag = dag(parents)

            def to_cpt(X):
                X = Dirichlet._ensure_moments(X, DirichletMoments)
                return np.broadcast_to(
                    np.exp(X.get_moments()[0]),
                    X.plates + X.dims[0]
                )


            def _check(X, y):
                X.update()
                cpts = {
                    name: to_cpt(config["table"])
                    for (name, config) in dag.items()
                }
                for (name, ind) in y.items():
                    cpts[name] = cpts[name] * onehot(
                        ind,
                        cpts[name].shape[-1],
                        extradims=len(dag[name].get("given", []))
                    )
                msgs = messages(cpts)
                assert len(msgs) == len(parents)
                for (parent, msg) in zip(parents, msgs):
                    self.assertMessage(parent._message_from_children(), [msg])
                return


            X = CategoricalGraph(dag, **kwargs)
            _check(X, {})
            for y in observations:
                X.observe(y)
                _check(X, y)

            return


        # Simple case
        _run(
            parents=[
                Dirichlet(np.random.rand(2)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                },
            },
            messages=lambda cpts: [
                normalize(cpts["x"])
            ],
            observations=[
                {"x": 1},
            ]
        )

        # Parent with plates
        _run(
            parents=[
                Dirichlet(np.random.rand(2, 3)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": parents[0],
                },
            },
            messages=lambda cpts: [
                normalize(np.einsum("x,xy->xy", cpts["x"], cpts["y"]))
            ],
            observations=[
                {"x": 1},
                {"y": 0},
                {"x": 0, "y": 0},
            ]
        )

        # Same parent in multiple CPTs
        _run(
            parents=[
                Dirichlet(np.random.rand(2)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                },
                "y": {
                    "table": parents[0],
                },
            },
            messages=lambda cpts: [
                normalize(cpts["x"]) + normalize(cpts["y"])
            ],
            observations=[
                {"x": 1},
                {"y": 1},
                {"x": 1, "y": 1},
            ]
        )

        # Multiple parents
        _run(
            parents=[
                Dirichlet(np.random.rand(3)),
                Dirichlet(np.random.rand(3, 4)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                },
                "y": {
                    "table": parents[1],
                    "given": ["x"],
                },
            },
            messages=lambda cpts: [
                sumproduct("x,xy->x", cpts["x"], cpts["y"]),
                sumproduct("x,xy->xy", cpts["x"], cpts["y"]),
            ],
            observations=[
                {"x": 0},
                {"y": 0},
                {"x": 0, "y": 0},
            ]
        )

        # DAG-plates not in parent
        _run(
            parents=[
                Dirichlet(np.random.rand(2, 3)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": parents[0],
                },
            },
            messages=lambda cpts: [
                np.sum(
                    normalize(
                        np.einsum(
                            "ax,axy->axy",
                            np.broadcast_to(cpts["x"], (5, 2)),
                            np.broadcast_to(cpts["y"], (5, 2, 3)),
                        ),
                        axis=(1,2)
                    ),
                    axis=0
                )
            ],
            observations=[
                {"x": [1, 1, 0, 0, 1]},
                {"y": [2, 1, 2, 0, 2]},
                {"x": [1, 1, 0, 0, 1], "y": [2, 1, 2, 0, 2]},
            ],
            plates=(5,)
        )

        # DAG-plates are in parent
        _run(
            parents=[
                Dirichlet(np.random.rand(5, 2, 3)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": random(2),
                },
                "y": {
                    "given": ["x"],
                    "table": parents[0],
                },
            },
            messages=lambda cpts: [
                normalize(
                    np.einsum(
                        "ax,axy->axy",
                        np.broadcast_to(cpts["x"], (5, 2)),
                        np.broadcast_to(cpts["y"], (5, 2, 3)),
                    ),
                    axis=(1,2)
                )
            ],
            observations=[
                {"x": [1, 1, 0, 0, 1]},
                {"y": [2, 1, 2, 0, 2]},
                {"x": [1, 1, 0, 0, 1], "y": [2, 1, 2, 0, 2]},
            ],
            plates=(5,)
        )

        pass


    def test_message_to_children(self):

        def _run(dag, messages, **kwargs):

            def to_cpt(X):
                return np.exp(
                    Dirichlet._ensure_moments(
                        X,
                        DirichletMoments
                    ).get_moments()[0]
                )


            def _check(Y, msg):
                m = Y.get_moments()
                assert len(m) == 1
                self.assertAllClose(m[0], msg)
                return


            X = CategoricalGraph(dag, **kwargs)
            X.update()
            cpts = {
                name: to_cpt(config["table"])
                for (name, config) in dag.items()
            }
            for (variable, msg) in messages(cpts).items():
                Y = X[variable]
                _check(Y, msg)

            return

        # Single variable
        #
        # x
        _run(
            {
                "x": {
                    "table": random(3),
                }
            },
            lambda cpts: {"x": normalize(cpts["x"])},
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
                "marg_y": sumproduct("x,xy->y", cpts["x"], cpts["y"]),
                "yx": sumproduct("x,xy->yx", cpts["x"], cpts["y"]),
            },
            marginals={
                "marg_y": ["y"],
                "yx": ["y", "x"],
            },
        )

        # Pair with plates
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
            lambda _cpts: (
                (
                    lambda cpts: {
                        "x": sumproduct("ax,axy->ax", cpts["x"], cpts["y"], plates_ndim=1),
                        "y": sumproduct("ax,axy->ay", cpts["x"], cpts["y"], plates_ndim=1),
                        "marg_y": sumproduct("ax,axy->ay", cpts["x"], cpts["y"], plates_ndim=1),
                        "yx": sumproduct("ax,axy->ayx", cpts["x"], cpts["y"], plates_ndim=1),
                    }
                )(
                    {
                        "x": np.broadcast_to(_cpts["x"], (5, 2)),
                        "y": np.broadcast_to(_cpts["y"], (5, 2, 3)),
                    }
                )
            ),
            marginals={
                "marg_y": ["y"],
                "yx": ["y", "x"],
            },
            plates=(5,)
        )

        pass


    def test_message_from_children(self):

        def _run(dag, messages, moments, **kwargs):

            class DummyNode():
                def __init__(self, parent, msg):
                    parent._add_child(self, 0)
                    self.msg = msg
                def _message_to_parent(self, index, u_parent=None):
                    return [self.msg]

            def to_cpt(X):
                return np.exp(
                    Dirichlet._ensure_moments(
                        X,
                        DirichletMoments
                    ).get_moments()[0]
                )

            def _check(Y, msg):
                m = Y.get_moments()
                assert len(m) == 1
                self.assertAllClose(m[0], msg)
                return

            X = CategoricalGraph(dag, **kwargs)

            for (variable, message) in messages:
                DummyNode(X[variable], np.log(message))

            X.update()
            cpts = {
                name: to_cpt(config["table"])
                for (name, config) in dag.items()
            }
            msgs = [msg for (_, msg) in messages]
            for (variable, msg) in moments(cpts, *msgs).items():
                Y = X[variable]
                _check(Y, msg)

            return

        _run(
            {
                "x": {
                    "table": [0.3, 0.6, 0.1],
                },
                "y": {
                    "given": ["x"],
                    "table": [ [0.9, 0.1], [0.5, 0.5], [0.1, 0.9] ],
                },
            },
            [
                ("y", [10, 1]),
            ],
            lambda cpts, m0: {
                "x": normalize(np.einsum("x,xy,y->x", cpts["x"], cpts["y"], m0)),
                "y": normalize(np.einsum("x,xy,y->y", cpts["x"], cpts["y"], m0)),
            }
        )

        _run(
            {
                "x": {
                    "table": [0.3, 0.6, 0.1],
                },
                "y": {
                    "given": ["x"],
                    "table": [ [0.9, 0.1], [0.5, 0.5], [0.1, 0.9] ],
                },
            },
            [
                ("y", [10, 1]),
                ("y", [30, 1]),
                ("x", [1, 20, 10]),
            ],
            lambda cpts, m0, m1, m2: {
                "x": normalize(np.einsum("x,xy,y,y,x->x", cpts["x"], cpts["y"], m0, m1, m2)),
                "y": normalize(np.einsum("x,xy,y,y,x->y", cpts["x"], cpts["y"], m0, m1, m2)),
                "xy": normalize(np.einsum("x,xy,y,y,x->xy", cpts["x"], cpts["y"], m0, m1, m2)),
            },
            marginals={
                "xy": ["x", "y"],
            }
        )

        # Test message from explicit marginal (with plates)
        _run(
            {
                "x": {
                    "table": [0.3, 0.6, 0.1],
                },
                "y": {
                    "given": ["x"],
                    "table": [ [0.9, 0.1], [0.5, 0.5], [0.1, 0.9] ],
                },
            },
            [
                ("y", [[10, 1], [4, 2], [6, 7], [9, 9], [1, 3]]),
                ("xy", [
                    [[10, 1], [3, 2], [1, 1]],
                    [[4, 2], [5, 10], [3, 9]],
                    [[6, 7], [12, 30], [4, 9]],
                    [[1, 1], [2, 2], [10, 9]],
                    [[9, 9], [1, 3], [11, 1]],
                ]),
            ],
            lambda cpts, m_y, m_xy: {
                "x": sumproduct("x,xy,ay,axy->ax", cpts["x"], cpts["y"], m_y, m_xy, plates_ndim=1),
                "y": sumproduct("x,xy,ay,axy->ay", cpts["x"], cpts["y"], m_y, m_xy, plates_ndim=1),
                "xy": sumproduct("x,xy,ay,axy->axy", cpts["x"], cpts["y"], m_y, m_xy, plates_ndim=1),
            },
            marginals={
                "xy": ["x", "y"],
            },
            plates=(5,)
        )

        return


    def test_message_from_marginal(self):

        X = CategoricalGraph(
            {
                "x": {
                    "table": [0.3, 0.5, 0.2]
                }
            },
            plates=(5,)
        )

        x_marg = X["x"]

        Y = Mixture(x_marg, Bernoulli, [0.7, 0.5, 0.9])

        Y.observe([1, 0, 0, 1, 1])

        assert len(X._message_from_children()) == 1
        self.assertAllClose(
            X._message_from_children()[0],
            np.log(
                [
                    [0.7, 0.5, 0.9],
                    [0.3, 0.5, 0.1],
                    [0.3, 0.5, 0.1],
                    [0.7, 0.5, 0.9],
                    [0.7, 0.5, 0.9],
                ]
            )
        )

        return


    def test_lower_bound_contribution(self):

        cpt_x = [0.3, 0.6, 0.1]
        cpt_y = [ [0.9, 0.1], [0.5, 0.5], [0.1, 0.9] ]
        X = CategoricalGraph(
            {
                "x": {
                    "table": cpt_x,
                },
                "y": {
                    "given": ["x"],
                    "table": cpt_y,
                },
            },
        )

        Y = Mixture(X["y"], GaussianARD, [-1, 1], 1)

        X.update()
        Y.update()

        # Without observations, there's no likelihood term
        np.testing.assert_allclose(
            X.lower_bound_contribution(),
            0.0,
        )

        # Test with observation node below the graph
        Y.observe(-2)
        X.update()

        np.testing.assert_allclose(
            Y.lower_bound_contribution() + X.lower_bound_contribution(),
            np.log(np.einsum(
                "x,xy,y->",
                # P(x)
                cpt_x,
                # P(y|x)
                cpt_y,
                # P(obs|y)
                np.exp([
                    bp_random.gaussian_logpdf(
                        (-2)**2, (-2)*(-1), (-1)**2, 0, 1
                    ),
                    bp_random.gaussian_logpdf(
                        (-2)**2, (-2)*1, 1**2, 0, 1
                    ),
                ])
            ))
        )

        return



    def test_integration_with_vb(self):

        dag = CategoricalGraph(
            {
                "z0": {
                    "table": [0.3, 0.6, 0.1],
                },
                "z1": {
                    "given": ["z0"],
                    "table": [ [0.9, 0.1], [0.5, 0.5], [0.1, 0.9] ],
                },
            }
        )

        # Create a mixture and data from it so that it is practically
        # "impossible" that the data came from the other cluster. That is, far
        # apart Gaussian clusters and data from the other.
        Y = Mixture(dag["z1"], GaussianARD, [100, -100], [1, 1], plates=(10,))
        Y.observe(-100 + np.random.randn(10))

        Q = VB(Y, dag)

        Q.update(repeat=10)

        np.testing.assert_allclose(
            dag["z1"].get_moments()[0],
            # Data had strong evidence for the second cluster
            [0, 1],
        )

        np.testing.assert_allclose(
            dag["z0"].get_moments()[0],
            np.array([0.3*0.1, 0.6*0.5, 0.1*0.9]) / (0.3*0.1 + 0.6*0.5 + 0.1*0.9)
        )

        return
