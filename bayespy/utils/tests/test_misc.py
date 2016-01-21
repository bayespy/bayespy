################################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for bayespy.utils.misc module.
"""

import unittest

import warnings

import numpy as np
from scipy.special import psi

from numpy import testing

from .. import misc

class TestCeilDiv(misc.TestCase):

    def test_ceildiv(self):
        """
        Test the ceil division
        """

        self.assertEqual(misc.ceildiv(3, 1),
                         3)
        
        self.assertEqual(misc.ceildiv(6, 3),
                         2)
        
        self.assertEqual(misc.ceildiv(7, 3),
                         3)
        
        self.assertEqual(misc.ceildiv(8, 3),
                         3)
        
        self.assertEqual(misc.ceildiv(-6, 3),
                         -2)
        
        self.assertEqual(misc.ceildiv(-7, 3),
                         -2)
        
        self.assertEqual(misc.ceildiv(-8, 3),
                         -2)
        
        self.assertEqual(misc.ceildiv(-9, 3),
                         -3)
        
        self.assertEqual(misc.ceildiv(6, -3),
                         -2)
        
        self.assertEqual(misc.ceildiv(7, -3),
                         -2)
        
        self.assertEqual(misc.ceildiv(8, -3),
                         -2)
        
        self.assertEqual(misc.ceildiv(9, -3),
                         -3)
        

class TestAddAxes(misc.TestCase):

    def test_add_axes(self):
        """
        Test the add_axes method.
        """
        f = lambda X, **kwargs: np.shape(misc.add_axes(X, **kwargs))

        # By default, add one leading axis
        self.assertEqual(f(np.ones((3,))),
                         (1,3))

        # By default, add leading axes
        self.assertEqual(f(np.ones((3,)), num=3),
                         (1,1,1,3))

        # By default, add one axis               
        self.assertEqual(f(np.ones((3,)), axis=1),
                         (3,1))
                         
        # Add axes to the beginning
        self.assertEqual(f(np.ones((2,3,4,)), axis=0, num=3),
                         (1,1,1,2,3,4))
        
        # Add axes to the middle
        self.assertEqual(f(np.ones((2,3,4,)), axis=1, num=3),
                         (2,1,1,1,3,4))

        # Test negative axis index
        self.assertEqual(f(np.ones((2,3,4,)), axis=-4, num=3),
                         (1,1,1,2,3,4))
        self.assertEqual(f(np.ones((2,3,4,)), axis=-1, num=1),
                         (2,3,4,1))
        self.assertEqual(f(np.ones((2,3,4,)), axis=-2, num=3),
                         (2,3,1,1,1,4))
                         
        # Add axes to the end
        self.assertEqual(f(np.ones((2,3,4,)), axis=3, num=3),
                         (2,3,4,1,1,1))


class TestBroadcasting(unittest.TestCase):

    def test_is_shape_subset(self):
        f = misc.is_shape_subset
        self.assertTrue(f( (), () ))
        self.assertTrue(f( (), (3,) ))
        self.assertTrue(f( (1,), (1,) ))
        self.assertTrue(f( (1,), (3,) ))
        self.assertTrue(f( (1,), (4,1) ))
        self.assertTrue(f( (1,), (4,3) ))
        self.assertTrue(f( (1,), (1,3) ))
        self.assertTrue(f( (3,), (1,3) ))
        self.assertTrue(f( (3,), (4,3) ))
        self.assertTrue(f( (5,1,3), (6,5,4,3) ))
        self.assertTrue(f( (5,4,3), (6,5,4,3) ))

        self.assertFalse(f( (1,), () ))
        self.assertFalse(f( (3,), (1,) ))
        self.assertFalse(f( (4,3,), (3,) ))
        self.assertFalse(f( (4,3,), (1,3,) ))
        self.assertFalse(f( (6,1,4,3,), (6,1,1,3,) ))

class TestMultiplyShapes(unittest.TestCase):

    def test_multiply_shapes(self):
        f = lambda *shapes: tuple(misc.multiply_shapes(*shapes))

        # Basic test
        self.assertEqual(f((2,),
                           (3,)),
                         (6,))
        # Test multiple arguments
        self.assertEqual(f((2,),
                           (3,),
                           (4,)),
                         (24,))
        # Test different lengths and multiple arguments
        self.assertEqual(f((  2,3,),
                           (4,5,6,),
                           (    7,)),
                         (4,10,126,))
        # Test empty shapes
        self.assertEqual(f((),
                           ()),
                         ())
        self.assertEqual(f((),
                           (5,)),
                         (5,))

class TestSumMultiply(unittest.TestCase):

    def check_sum_multiply(self, *shapes, **kwargs):

        # The set of arrays
        x = list()
        for (ind, shape) in enumerate(shapes):
            x += [np.random.randn(*shape)]

        # Result from the function
        yh = misc.sum_multiply(*x,
                               **kwargs)

        axis = kwargs.get('axis', None)
        sumaxis = kwargs.get('sumaxis', True)
        keepdims = kwargs.get('keepdims', False)
        
        # Compute the product
        y = 1
        for xi in x:
            y = y * xi

        # Compute the sum
        if sumaxis:
            y = np.sum(y, axis=axis, keepdims=keepdims)
        else:
            axes = np.arange(np.ndim(y))
            # TODO/FIXME: np.delete has a bug that it doesn't accept negative
            # indices. Thus, transform negative axes to positive axes.
            if len(axis) > 0:
                axis = [i if i >= 0 
                        else i+np.ndim(y) 
                        for i in axis]
            elif axis < 0:
                axis += np.ndim(y)
                
            axes = np.delete(axes, axis)
            axes = tuple(axes)
            if len(axes) > 0:
                y = np.sum(y, axis=axes, keepdims=keepdims)

        # Check the result
        testing.assert_allclose(yh, y,
                                err_msg="Incorrect value.")
        

    def test_sum_multiply(self):
        """
        Test misc.sum_multiply.
        """
        # Check empty list returns error
        self.assertRaises(ValueError, 
                          self.check_sum_multiply)
        # Check scalars
        self.check_sum_multiply(())
        self.check_sum_multiply((), (), ())
        
        # Check doing no summation
        self.check_sum_multiply((3,), 
                                axis=())
        self.check_sum_multiply((3,1,5), 
                                (  4,1),
                                (    5,),
                                (      ),
                                axis=(), 
                                keepdims=True)
        # Check AXES_TO_SUM
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                axis=(1,))
        self.check_sum_multiply((3,1),
                                (1,4),
                                (3,4),
                                axis=(-2,))
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                axis=(1,-2))
        # Check AXES_TO_SUM and KEEPDIMS
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                axis=(1,),
                                keepdims=True)
        self.check_sum_multiply((3,1),
                                (1,4),
                                (3,4),
                                axis=(-2,),
                                keepdims=True)
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                axis=(1,-2,),
                                keepdims=True)
        self.check_sum_multiply((3,1,5,6), 
                                (  4,1,6),
                                (  4,1,1),
                                (       ),
                                axis=(1,-2),
                                keepdims=True)
        # Check AXES_TO_KEEP
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(1,))
        self.check_sum_multiply((3,1),
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(-2,))
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(1,-2))
        # Check AXES_TO_KEEP and KEEPDIMS
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(1,),
                                keepdims=True)
        self.check_sum_multiply((3,1),
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(-2,),
                                keepdims=True)
        self.check_sum_multiply((3,1), 
                                (1,4),
                                (3,4),
                                sumaxis=False,
                                axis=(1,-2,),
                                keepdims=True)
        self.check_sum_multiply((3,1,5,6), 
                                (  4,1,6),
                                (  4,1,1),
                                (       ),
                                sumaxis=False,
                                axis=(1,-2,),
                                keepdims=True)
        # Check errors
        # Inconsistent shapes
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,5))
        # Axis index out of bounds
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          axis=(-3,))
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          axis=(2,))
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          sumaxis=False,
                          axis=(-3,))
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          sumaxis=False,
                          axis=(2,))
        # Same axis several times
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          axis=(1,-1))
        self.assertRaises(ValueError, 
                          self.check_sum_multiply,
                          (3,4),
                          (3,4),
                          sumaxis=False,
                          axis=(1,-1))

class TestLogSumExp(misc.TestCase):

    def test_logsumexp(self):
        """
        Test the ceil division
        """

        self.assertAllClose(misc.logsumexp(3),
                            np.log(np.sum(np.exp(3))))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAllClose(misc.logsumexp(-np.inf),
                                -np.inf)

        self.assertAllClose(misc.logsumexp(np.inf),
                            np.inf)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAllClose(misc.logsumexp(np.nan),
                                np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAllClose(misc.logsumexp([-np.inf, -np.inf]),
                                -np.inf)

        self.assertAllClose(misc.logsumexp([[1e10,  1e-10],
                                             [-1e10, -np.inf]], axis=-1),
                            [1e10, -1e10])

        # Test keeping dimensions
        self.assertAllClose(misc.logsumexp([[1e10,  1e-10],
                                             [-1e10, -np.inf]], 
                                            axis=-1,
                                            keepdims=True),
                            [[1e10], [-1e10]])

        # Test multiple axes
        self.assertAllClose(misc.logsumexp([[1e10,  1e-10],
                                             [-1e10, -np.inf]], 
                                            axis=(-1,-2)),
                            1e10)

        pass

class TestMean(misc.TestCase):

    def test_mean(self):
        """
        Test the ceil division
        """

        self.assertAllClose(misc.mean(3),
                            3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAllClose(misc.mean(np.nan),
                                np.nan)

            self.assertAllClose(misc.mean([[2,3],
                                           [np.nan,np.nan]], 
                                           axis=-1),
                                [2.5,np.nan])
            self.assertAllClose(misc.mean([[2,3],
                                           [np.nan,np.nan]], 
                                           axis=-1,
                                           keepdims=True),
                                [[2.5],[np.nan]])
        
        self.assertAllClose(misc.mean([[2,3],
                                       [np.nan,np.nan]], 
                                       axis=-2),
                            [2,3])
        self.assertAllClose(misc.mean([[2,3],
                                       [np.nan,np.nan]], 
                                       axis=-2,
                                       keepdims=True),
                            [[2,3]])

        self.assertAllClose(misc.mean([[2,3],
                                       [np.nan,np.nan]]),
                            2.5)
        self.assertAllClose(misc.mean([[2,3],
                                       [np.nan,np.nan]],
                                       axis=(-1,-2)),
                            2.5)
        self.assertAllClose(misc.mean([[2,3],
                                       [np.nan,np.nan]], 
                                       keepdims=True),
                            [[2.5]])
        
        pass


class TestInvPsi(misc.TestCase):

    def test_invpsi(self):
        x = 1000
        y = psi(x)
        self.assertAllClose(misc.invpsi(y), x)

        x = 1/1000
        y = psi(x)
        self.assertAllClose(misc.invpsi(y), x, rtol=1e-3)

        x = 50*np.random.rand(5)
        y = psi(x)
        self.assertAllClose(misc.invpsi(y), x)

        pass


class TestPutSimple(misc.TestCase):

    def test_put_simple(self):

        # Scalar indices
        self.assertAllClose(
            misc.put_simple(
                42,
                2,
            ),
            [0, 0, 42],
        )

        # Simple vectors, automatic length
        self.assertAllClose(
            misc.put_simple(
                [1, 0.1, 0.01, 0.001, 0.0001],
                [3, 3, 1, 3, 0],
            ),
            [0.0001, 0.01, 0, 1.101],
        )

        # Matrix indices
        self.assertAllClose(
            misc.put_simple(
                [[1, 0.1], [0.01, 0.001]],
                [[4, 1], [1, 3]],
            ),
            [0, 0.11, 0, 0.001, 1],
        )

        # Test axis
        self.assertAllClose(
            misc.put_simple(
                [[1, 0.1], [0.01, 0.001], [0.0001, 0.00001]],
                [3, 3, 0],
                axis=-2,
            ),
            [[0.0001, 0.00001], [0, 0], [0, 0], [1.01, 0.101]],
        )

        # Test explicit length
        self.assertAllClose(
            misc.put_simple(
                [1, 0.1, 0.01, 0.001, 0.0001],
                [3, 3, 1, 3, 0],
                length=6,
            ),
            [0.0001, 0.01, 0, 1.101, 0, 0],
        )

        # Test broadcasting
        self.assertAllClose(
            misc.put_simple(
                2,
                [3, 3, 1, 3, 0],
            ),
            [2, 2, 0, 6],
        )

        # Test leading axes in y
        self.assertAllClose(
            misc.put_simple(
                [[1, 0.1], [0.01, 0.001], [0.0001, 0.00001]],
                [2, 0],
            ),
            [[0.1, 0, 1], [0.001, 0, 0.01], [0.00001, 0, 0.0001]],
        )

        pass
