################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `take` module.
"""

import numpy as np

from bayespy.nodes import GaussianARD
from bayespy.nodes import Take

from bayespy.inference import VB

from bayespy.utils.misc import TestCase


class TestTake(TestCase):


    def test_parent_validity(self):
        """
        Test that the parent nodes are validated properly
        """

        # Test scalar index, no shape
        X = GaussianARD(1, 1, plates=(2,), shape=())
        Y = Take(X, 1)
        self.assertEqual(
            Y.plates,
            (),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test vector indices, no shape
        X = GaussianARD(1, 1, plates=(2,), shape=())
        Y = Take(X, [1, 1, 0, 1])
        self.assertEqual(
            Y.plates,
            (4,),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test matrix indices, no shape
        X = GaussianARD(1, 1, plates=(2,), shape=())
        Y = Take(X, [[1, 1, 0], [1, 0, 1]])
        self.assertEqual(
            Y.plates,
            (2, 3),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test scalar index, with shape
        X = GaussianARD(1, 1, plates=(3,), shape=(2,))
        Y = Take(X, 2)
        self.assertEqual(
            Y.plates,
            (),
        )
        self.assertEqual(
            Y.dims,
            ( (2,), (2, 2) )
        )

        # Test vector indices, with shape
        X = GaussianARD(1, 1, plates=(3,), shape=(2,))
        Y = Take(X, [1, 1, 0, 2])
        self.assertEqual(
            Y.plates,
            (4,),
        )
        self.assertEqual(
            Y.dims,
            ( (2,), (2, 2) )
        )

        # Test matrix indices, no shape
        X = GaussianARD(1, 1, plates=(3,), shape=(2,))
        Y = Take(X, np.ones((4, 5), dtype=np.int))
        self.assertEqual(
            Y.plates,
            (4, 5),
        )
        self.assertEqual(
            Y.dims,
            ( (2,), (2, 2) )
        )

        # Test scalar indices with more plate axes
        X = GaussianARD(1, 1, plates=(4, 2), shape=())
        Y = Take(X, 1)
        self.assertEqual(
            Y.plates,
            (4,),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test vector indices with more plate axes
        X = GaussianARD(1, 1, plates=(4, 2), shape=())
        Y = Take(X, np.ones(3, dtype=np.int))
        self.assertEqual(
            Y.plates,
            (4, 3),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test take on other plate axis
        X = GaussianARD(1, 1, plates=(4, 2), shape=())
        Y = Take(X, np.ones(3, dtype=np.int), plate_axis=-2)
        self.assertEqual(
            Y.plates,
            (3, 2),
        )
        self.assertEqual(
            Y.dims,
            ( (), () )
        )

        # Test positive plate axis
        X = GaussianARD(1, 1, plates=(4, 2), shape=())
        self.assertRaises(
            ValueError,
            Take,
            X,
            np.ones(3, dtype=np.int),
            plate_axis=0,
        )

        # Test indices out of bounds
        X = GaussianARD(1, 1, plates=(2,), shape=())
        self.assertRaises(
            ValueError,
            Take,
            X,
            [0, -3],
        )
        X = GaussianARD(1, 1, plates=(2,), shape=())
        self.assertRaises(
            ValueError,
            Take,
            X,
            [0, 2],
        )

        # Test non-integer indices
        X = GaussianARD(1, 1, plates=(2,), shape=())
        self.assertRaises(
            ValueError,
            Take,
            X,
            [0, 1.5],
        )

        pass


    def test_moments(self):
        """
        Test moments computation in Take node
        """

        # Test scalar index, no shape
        X = GaussianARD([1, 2], [1, 0.5], shape=())
        Y = Take(X, 1)
        self.assertAllClose(
            Y.get_moments()[0],
            2,
        )
        self.assertAllClose(
            Y.get_moments()[1],
            6,
        )

        # Test vector indices, no shape
        X = GaussianARD([1, 2], [1, 0.5], shape=())
        Y = Take(X, [1, 1, 0, 1])
        self.assertAllClose(
            Y.get_moments()[0],
            [2, 2, 1, 2],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [6, 6, 2, 6],
        )

        # Test matrix indices, no shape
        X = GaussianARD([1, 2], [1, 0.5], shape=())
        Y = Take(X, [[1, 1, 0], [1, 0, 1]])
        self.assertAllClose(
            Y.get_moments()[0],
            [[2, 2, 1], [2, 1, 2]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [[6, 6, 2], [6, 2, 6]],
        )

        # Test scalar index, with shape
        X = GaussianARD([[1, 2], [3, 4], [5, 6]], [[1, 1/2], [1/3, 1/4], [1/5, 1/6]], shape=(2,))
        Y = Take(X, 2)
        self.assertAllClose(
            Y.get_moments()[0],
            [5, 6],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [[25+5, 30], [30, 36+6]],
        )

        # Test vector indices, with shape
        X = GaussianARD([[1, 2], [3, 4], [5, 6]], [[1, 1/2], [1/3, 1/4], [1/5, 1/6]], shape=(2,))
        Y = Take(X, [1, 1, 0, 2])
        self.assertAllClose(
            Y.get_moments()[0],
            [[3, 4], [3, 4], [1, 2], [5, 6]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [
                [[9+3, 12], [12, 16+4]],
                [[9+3, 12], [12, 16+4]],
                [[1+1, 2], [2, 4+2]],
                [[25+5, 30], [30, 36+6]]
            ],
        )

        # Test matrix indices, no shape
        X = GaussianARD([[1, 2], [3, 4], [5, 6]], [[1, 1/2], [1/3, 1/4], [1/5, 1/6]], shape=(2,))
        Y = Take(X, [[1, 1], [0, 2]])
        self.assertAllClose(
            Y.get_moments()[0],
            [[[3, 4], [3, 4]], [[1, 2], [5, 6]]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [
                [[[9+3, 12], [12, 16+4]],
                 [[9+3, 12], [12, 16+4]]],
                [[[1+1, 2], [2, 4+2]],
                 [[25+5, 30], [30, 36+6]]],
            ],
        )

        # Test with more plate axes
        X = GaussianARD([[1, 2], [3, 4], [5, 6]], [[1, 1/2], [1/3, 1/4], [1/5, 1/6]], shape=())
        Y = Take(X, [1, 0, 1])
        self.assertAllClose(
            Y.get_moments()[0],
            [[2, 1, 2], [4, 3, 4], [6, 5, 6]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [[4+2, 1+1, 4+2], [16+4, 9+3, 16+4], [36+6, 25+5, 36+6]],
        )

        # Test take on other plate axis
        X = GaussianARD([[1, 2], [3, 4], [5, 6]], [[1, 1/2], [1/3, 1/4], [1/5, 1/6]], shape=())
        Y = Take(X, [2, 0], plate_axis=-2)
        self.assertAllClose(
            Y.get_moments()[0],
            [[5, 6], [1, 2]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [[25+5, 36+6], [1+1, 4+2]],
        )

        # Test parent broadcasting
        X = GaussianARD([1, 2], [1, 1/2], plates=(3,), shape=(2,))
        Y = Take(X, [1, 1, 0, 1])
        self.assertAllClose(
            Y.get_moments()[0],
            [[1, 2], [1, 2], [1, 2], [1, 2]],
        )
        self.assertAllClose(
            Y.get_moments()[1],
            [
                [[1+1, 2], [2, 4+2]],
                [[1+1, 2], [2, 4+2]],
                [[1+1, 2], [2, 4+2]],
                [[1+1, 2], [2, 4+2]],
            ]
        )

        pass


    def test_message_to_parent(self):
        """
        Test parent message computation in Take node
        """

        def check(indices, plates, shape, axis=-1, use_mask=False):
            mu = np.random.rand(*(plates+shape))
            alpha = np.random.rand(*(plates+shape))
            X = GaussianARD(mu, alpha, shape=shape, plates=plates)
            Y = Take(X, indices, plate_axis=axis)
            Z = GaussianARD(Y, 1, shape=shape)
            z = np.random.randn(*(Z.get_shape(0)))
            if use_mask:
                mask = np.mod(np.reshape(np.arange(np.prod(Z.plates)), Z.plates), 2) != 0
            else:
                mask = True
            Z.observe(z, mask=mask)
            X.update()
            (x0, x1) = X.get_moments()

            # For comparison, build the same model brute force
            X = GaussianARD(mu, alpha, shape=shape, plates=plates)

            # Number of trailing plate axes before the take axis
            N = len(X.plates) + axis

            # Reshape the take axes into a single axis
            z_shape = X.plates[:axis] + (-1,)
            if axis < -1:
                z_shape = z_shape + X.plates[(axis+1):]
            z_shape = z_shape + shape
            z = np.reshape(z, z_shape)

            # Reshape the take axes into a single axis
            if use_mask:
                mask_shape = X.plates[:axis] + (-1,)
                if axis < -1:
                    mask_shape = mask_shape + X.plates[(axis+1):]
                mask = np.reshape(mask, mask_shape)

            for (j, i) in enumerate(range(np.size(indices))):
                ind = np.array(indices).flatten()[i]
                index_x = N*(slice(None),) + (ind,)
                index_z = N*(slice(None),) + (j,)
                # print(index)
                Xi = X[index_x]
                zi = z[index_z]
                Zi = GaussianARD(Xi, 1, ndim=len(shape))
                if use_mask:
                    maski = mask[index_z]
                else:
                    maski = True
                Zi.observe(zi, mask=maski)

            X.update()

            self.assertAllClose(
                x0,
                X.get_moments()[0],
            )

            self.assertAllClose(
                x1,
                X.get_moments()[1],
            )

            return


        # Test scalar index
        check(1, (2,), ())
        check(1, (2, 3), ())
        check(1, (2, 3), (4,))
        check(1, (2, 3), (), axis=-2)
        check(1, (2, 3), (4,), axis=-2)
        check(1, (2,), (), use_mask=True)
        check(1, (2, 3), (), use_mask=True)
        check(1, (2, 3), (4,), use_mask=True)
        check(1, (2, 3), (), axis=-2, use_mask=True)
        check(1, (2, 3), (4,), axis=-2, use_mask=True)

        # Test vector index
        check([1, 1, 0, 1], (2,), ())
        check([1, 1, 0, 1], (2, 3), ())
        check([1, 1, 0, 1], (2, 3), (4,))
        check([1, 1, 0, 1], (2, 3), (), axis=-2)
        check([1, 1, 0, 1], (2, 3), (4,), axis=-2)
        check([1, 1, 0, 1], (2,), (), use_mask=True)
        check([1, 1, 0, 1], (2, 3), (), use_mask=True)
        check([1, 1, 0, 1], (2, 3), (4,), use_mask=True)
        check([1, 1, 0, 1], (2, 3), (), axis=-2, use_mask=True)
        check([1, 1, 0, 1], (2, 3), (4,), axis=-2, use_mask=True)

        # Test matrix index
        check([[1, 1, 2], [0, 2, 1]], (4,), ())
        check([[1, 1, 2], [0, 2, 1]], (4, 5), ())
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (6,))
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (), axis=-2)
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (6,), axis=-2)
        check([[1, 1, 2], [0, 2, 1]], (4,), (), use_mask=True)
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (), use_mask=True)
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (6,), use_mask=True)
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (), axis=-2, use_mask=True)
        check([[1, 1, 2], [0, 2, 1]], (4, 5), (6,), axis=-2, use_mask=True)

        pass
