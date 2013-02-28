######################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

"""
Unit tests for bayespy.utils.utils module.
"""

import unittest

import numpy as np

from numpy import testing

from .. import utils
#from bayespy.utils import utils

class TestBroadcasting(unittest.TestCase):

    def test_is_shape_subset(self):
        f = utils.is_shape_subset
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
        f = lambda *shapes: tuple(utils.multiply_shapes(*shapes))

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
        yh = utils.sum_multiply(*x,
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
        Test utils.sum_multiply.
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

class TestBandedSolve(unittest.TestCase):

    def test_block_banded_solve(self):
        """
        Test the Gaussian elimination algorithm for block-banded matrices.
        """

        #
        # Create a block-banded matrix
        #

        # Number of blocks
        N = 40

        # Random sizes of the blocks
        #D = np.random.randint(5, 10, size=N)
        # Fixed sizes of the blocks
        D = 5*np.ones(N)

        # Some helpful variables to create the covariances
        W = [np.random.randn(D[i], 2*D[i])
             for i in range(N)]

        # The diagonal blocks (covariances)
        A = [np.dot(W[i], W[i].T) for i in range(N)]
        # The superdiagonal blocks (cross-covariances)
        B = [np.dot(W[i][:,-1:], W[i+1][:,:1].T) for i in range(N-1)]

        C = utils.block_banded(A, B)

        # Create the system to be solved: y=C*x
        x_true = np.random.randn(np.sum(D), 3)
        y = np.dot(C, x_true)
        x_true = np.reshape(x_true, (N, -1, 3))
        y = np.reshape(y, (N, -1, 3))

        #
        # Run tests
        #

        # The correct inverse
        invC = np.linalg.inv(C)

        # Inverse from the function that is tested
        (invA, invB, x, ldet) = utils.block_banded_solve(A,B,y)

        #print("inv(C)=", invC)
        #print("blocks=", invA, invB)

        # Check that you get the correct number of blocks
        self.assertEqual(len(invA), N)
        self.assertEqual(len(invB), N-1)

        # Check each block
        i0 = 0
        for i in range(N-1):
            i1 = i0 + D[i]
            i2 = i1 + D[i+1]
            # Check diagonal block
            self.assertTrue(np.allclose(invA[i], invC[i0:i1, i0:i1]))
            # Check super-diagonal block
            self.assertTrue(np.allclose(invB[i], invC[i0:i1, i1:i2]))
            i0 = i1
        # Check last block
        self.assertTrue(np.allclose(invA[-1], invC[i0:, i0:]))

        # Check the solution of the system
        #print(x_true, x)
        self.assertTrue(np.allclose(x_true, x))

        # Check the log determinant
        self.assertAlmostEqual(ldet/np.linalg.slogdet(C)[1], 1)

## if __name__ == '__main__':
##     unittest.main()
