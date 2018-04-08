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

from ..categorical_graph import take, onehot

from bayespy.utils import random
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


    def test_take(self):

        # Scalar indices
        self.assertAllClose(
            take([1,2,3], 0, axis=-1),
            [1]
        )
        self.assertAllClose(
            take([[1,2,3], [4,5,6]], 2, axis=-1),
            [[3], [6]]
        )
        self.assertAllClose(
            take([[1,2,3], [4,5,6]], 1, axis=-2),
            [[4,5,6]]
        )

        # Array indices
        self.assertAllClose(
            take([[1,2,3], [4,5,6]], [1, 2], axis=-1),
            [[2], [6]]
        )
        self.assertAllClose(
            take(
                [
                    [[1,2,3,4], [4,5,6,7], [8,9,10,11]],
                    [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
                ],
                [1, 2],
                axis=-1
            ),
            [ [[2],[5],[9]], [[14],[18],[22]] ]
        )

        # Error: shape mismatches

        # Error: index out of bounds

        # Error: axis out of bounds

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
                "y": sumproduct("x,xy->xy", cpts["x"], cpts["y"]),
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
                "y": sumproduct("x,xy,yz->xy", cpts["x"], cpts["y"], cpts["z"]),
                "z": sumproduct("x,xy,yz->yz", cpts["x"], cpts["y"], cpts["z"]),
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
                "y": sumproduct("x,xy,xz,yzv->xy", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "z": sumproduct("x,xy,xz,yzv->xz", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "v": sumproduct("x,xy,xz,yzv->yzv", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
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
                "y": sumproduct("x,xy,yz,xzv->xy", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "z": sumproduct("x,xy,yz,xzv->yz", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
                "v": sumproduct("x,xy,yz,xzv->xzv", cpts["x"], cpts["y"], cpts["z"], cpts["v"]),
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
                "y": sumproduct("x,xy,yz->xy", cpts["x"], cpts["y"], cpts["z"]),
                "z": sumproduct("x,xy,yz->yz", cpts["x"], cpts["y"], cpts["z"]),
                "marg_y": sumproduct("x,xy,yz->y", cpts["x"], cpts["y"], cpts["z"]),
                "marg_z": sumproduct("x,xy,yz->z", cpts["x"], cpts["y"], cpts["z"]),
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
                "marg_y": {
                    "variables": ["y"],
                },
                "marg_z": {
                    "variables": ["z"],
                },
                "marg_xz": {
                    "variables": ["x", "z"],
                },
            }
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
            lambda cpts: {"x": normalize(cpts["x"], axis=-1)},
            [
                {"x": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
            ],
            plates={"a": 10},
        )

        # Pair with common plates
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(10, 2),
                    "plates": ["a"],
                },
                "y": {
                    "table": random(10, 2, 3),
                    "given": ["x"],
                    "plates": ["a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("ax,axy->ax", cpts["x"], cpts["y"], plates_ndim=1),
                "y": sumproduct("ax,axy->axy", cpts["x"], cpts["y"], plates_ndim=1),
            },
            [
                {"x": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
                {"y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
                {"x": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0], "y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
            ],
            plates={"a": 10},
        )

        # Pair with plates on children only
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
                    "table": random(10, 2, 3),
                    "given": ["x"],
                    "plates": ["a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("x,axy->x", cpts["x"], cpts["y"], plates_ndim=0),
                "y": sumproduct("x,axy->axy", cpts["x"], cpts["y"], plates_ndim=1),
            },
            [
                {"x": 1},
                {"y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
                {"x": 1, "y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
            ],
            plates={"a": 10},
        )

        # Broadcast plates
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
                    "table": random(2, 3),
                    "given": ["x"],
                    "plates": ["a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("x,axy->x", cpts["x"], np.broadcast_to(cpts["y"], (10,2,3)), plates_ndim=0),
                "y": sumproduct("x,axy->axy", cpts["x"], np.broadcast_to(cpts["y"], (10,2,3)), plates_ndim=1),
            },
            [
                {"x": 1},
                {"y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
                {"x": 1, "y": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
            ],
            plates={"a": 10},
        )

        # Pair with common and child-only plates
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(3, 2),
                    "plates": ["a"],
                },
                "y": {
                    "table": random(4, 3, 2, 5),
                    "given": ["x"],
                    "plates": ["b", "a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("ax,baxy->ax", cpts["x"], cpts["y"], plates_ndim=1),
                "y": sumproduct("ax,baxy->baxy", cpts["x"], cpts["y"], plates_ndim=2),
            },
            [
                {"x": [1, 1, 0]},
                {"y": [ [2, 1, 0], [2, 2, 4], [0, 0, 2], [4, 1, 2] ]},
                {"x": [1, 1, 0], "y": [ [4, 1, 0], [2, 3, 1], [0, 4, 2], [3, 1, 2] ]},
            ],
            plates={"a": 3, "b": 4},
        )

        # Pair with only common plates mapped with crossing
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(3, 4, 2),
                    "plates": ["a", "b"],
                },
                "y": {
                    "table": random(4, 3, 2, 5),
                    "given": ["x"],
                    "plates": ["b", "a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("abx,baxy->abx", cpts["x"], cpts["y"], plates_ndim=2),
                "y": sumproduct("abx,baxy->baxy", cpts["x"], cpts["y"], plates_ndim=2),
            },
            [
                {"x": [ [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0] ]},
                {"y": [ [2, 1, 0], [2, 2, 4], [0, 0, 2], [4, 1, 2] ]},
                {
                    "x": [ [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0] ],
                    "y": [ [2, 1, 0], [2, 2, 4], [0, 0, 2], [4, 1, 2] ]
                },
            ],
            plates={"a": 3, "b": 4},
        )

        # Plates from both parents
        #
        # x   y
        #  \ /
        #   z
        _run(
            {
                "x": {
                    "table": random(6, 5, 2),
                    "plates": ["b", "a"],
                },
                "y": {
                    "table": random(5, 4, 3),
                    "plates": ["a", "c"],
                },
                "z": {
                    "table": random(4, 5, 6, 2, 3, 7),
                    "plates": ["c", "a", "b"],
                    "given": ["x", "y"],
                },
            },
            lambda cpts: {
                "x": sumproduct("bax,acy,cabxyz->bax", cpts["x"], cpts["y"], cpts["z"], plates_ndim=2),
                "y": sumproduct("bax,acy,cabxyz->acy", cpts["x"], cpts["y"], cpts["z"], plates_ndim=2),
                "z": sumproduct("bax,acy,cabxyz->cabxyz", cpts["x"], cpts["y"], cpts["z"], plates_ndim=3),
            },
            [
                {"x": np.random.randint(2, size=(6,5))},
                {"y": np.random.randint(3, size=(5,4))},
                {"z": np.random.randint(7, size=(4,5,6))},
                {
                    "x": np.random.randint(2, size=(6,5)),
                    "y": np.random.randint(3, size=(5,4))
                },
                {
                    "x": np.random.randint(2, size=(6,5)),
                    "z": np.random.randint(7, size=(4,5,6)),
                },
                {
                    "y": np.random.randint(3, size=(5,4)),
                    "z": np.random.randint(7, size=(4,5,6)),
                },
                {
                    "x": np.random.randint(2, size=(6,5)),
                    "y": np.random.randint(3, size=(5,4)),
                    "z": np.random.randint(7, size=(4,5,6)),
                },
            ],
            plates={"a": 5, "b": 6, "c": 4},
        )

        # With custom marginals
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(3, 4, 2),
                    "plates": ["a", "b"],
                },
                "y": {
                    "table": random(4, 3, 2, 5),
                    "given": ["x"],
                    "plates": ["b", "a"],
                },
            },
            lambda cpts: {
                "x": sumproduct("abx,baxy->abx", cpts["x"], cpts["y"], plates_ndim=2),
                "y": sumproduct("abx,baxy->baxy", cpts["x"], cpts["y"], plates_ndim=2),
                "marg_y": sumproduct("abx,baxy->aby", cpts["x"], cpts["y"], plates_ndim=2),
            },
            [
                {"x": [ [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0] ]},
                {"y": [ [2, 1, 0], [2, 2, 4], [0, 0, 2], [4, 1, 2] ]},
                {
                    "x": [ [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0] ],
                    "y": [ [2, 1, 0], [2, 2, 4], [0, 0, 2], [4, 1, 2] ]
                },
            ],
            plates={"a": 3, "b": 4},
            marginals={
                "marg_y": {
                    "variables": ["y"],
                    # test swapping the order of plates compared to CPT
                    "plates": ["a", "b"],
                },
            },
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

        def _run(parents, dag, messages, observations, **kwargs):

            # Construct the DAG
            dag = dag(parents)

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

        # Child has plates
        _run(
            parents=[
                Dirichlet(np.random.rand(2)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                    "plates": ["trials"],
                },
            },
            messages=lambda cpts: [
                np.einsum("ax->x", normalize(np.broadcast_to(cpts["x"], (10, 2)), axis=-1)),
            ],
            observations=[
                {"x": np.ones(10, dtype=np.int)},
            ],
            plates={
                "trials": 10,
            },
        )

        # Both have plates
        _run(
            parents=[
                Dirichlet(np.random.rand(10, 2), plates=(10,)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                    "plates": ["trials"],
                },
            },
            messages=lambda cpts: [
                normalize(np.broadcast_to(cpts["x"], (10, 2)), axis=-1),
            ],
            observations=[
                {"x": np.ones(10, dtype=np.int)},
            ],
            plates={
                "trials": 10,
            },
        )

        # Both have plates but parent is currently broadcasting them
        _run(
            parents=[
                Dirichlet(np.random.rand(2), plates=(10,)),
            ],
            dag=lambda parents: {
                "x": {
                    "table": parents[0],
                    "plates": ["trials"],
                },
            },
            messages=lambda cpts: [
                normalize(np.broadcast_to(cpts["x"], (10, 2)), axis=-1),
            ],
            observations=[
                {"x": np.ones(10, dtype=np.int)},
            ],
            plates={
                "trials": 10,
            },
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


            def _check(Y, msg, plates):
                assert Y.plates == plates
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
            for (variable, (msg, plates)) in messages(cpts).items():
                Y = X[variable]
                _check(Y, msg, plates)

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
            lambda cpts: {"x": (normalize(cpts["x"]), ())},
        )
        _run(
            {
                "x": {
                    "table": random(3),
                    "plates": ["trials"],
                }
            },
            lambda cpts: {
                "x": (
                    np.broadcast_to(normalize(cpts["x"]), (10, 3)),
                    (10,)
                )
            },
            plates={"trials": 10},
        )

        # Pair
        #
        # x
        # |
        # y
        _run(
            {
                "x": {
                    "table": random(10, 2),
                    "plates": ["a"]
                },
                "y": {
                    "given": ["x"],
                    "table": random(10, 20, 2, 3),
                    "plates": ["a", "b"],
                },
            },
            lambda cpts: {
                "x": (
                    sumproduct("ax,abxy->ax", cpts["x"], cpts["y"], plates_ndim=1),
                    (10,)
                ),
                "y": (
                    sumproduct("ax,abxy->abxy", cpts["x"], cpts["y"], plates_ndim=2),
                    (10, 20)
                ),
                "marg_y": (
                    sumproduct("ax,abxy->bay", cpts["x"], cpts["y"], plates_ndim=2),
                    (20, 10)
                ),
            },
            marginals={
                "marg_y": {
                    "variables": ["y"],
                    "plates": ["b", "a"],
                }
            },
            plates={
                "a": 10,
                "b": 20,
            }
        )

        pass


    def test_message_from_children(self):
        pass
