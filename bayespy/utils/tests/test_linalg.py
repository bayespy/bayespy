################################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for bayespy.utils.linalg module.
"""

import numpy as np

from .. import misc
from .. import linalg

class TestDot(misc.TestCase):

    def test_dot(self):
        """
        Test dot product multiple multi-dimensional arrays.
        """

        # If no arrays, return 0
        self.assertAllClose(linalg.dot(),
                            0)
        # If only one array, return itself
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]]),
                            [[1,2,3],
                             [4,5,6]])
        # Basic test of two arrays: (2,3) * (3,2)
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]],
                                       [[7,8],
                                        [9,1],
                                        [2,3]]),
                            [[31,19],
                             [85,55]])
        # Basic test of four arrays: (2,3) * (3,2) * (2,1) * (1,2)
        self.assertAllClose(linalg.dot([[1,2,3],
                                        [4,5,6]],
                                       [[7,8],
                                        [9,1],
                                        [2,3]],
                                       [[4],
                                        [5]],
                                       [[6,7]]),
                            [[1314,1533],
                             [3690,4305]])

        # Test broadcasting: (2,2,2) * (2,2,2,2)
        self.assertAllClose(linalg.dot([[[1,2],
                                         [3,4]],
                                        [[5,6],
                                         [7,8]]],
                                       [[[[1,2],
                                          [3,4]],
                                         [[5,6],
                                          [7,8]]],
                                        [[[9,1],
                                          [2,3]],
                                         [[4,5],
                                          [6,7]]]]),
                            [[[[  7,  10],
                               [ 15,  22]],

                              [[ 67,  78],
                               [ 91, 106]]],


                             [[[ 13,   7],
                               [ 35,  15]],

                              [[ 56,  67],
                               [ 76,  91]]]])

        # Inconsistent shapes: (2,3) * (2,3)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[1,2,3],
                           [4,5,6]],
                          [[1,2,3],
                           [4,5,6]])
        # Other axes do not broadcast: (2,2,2) * (3,2,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[[1,2],
                            [3,4]],
                           [[5,6],
                            [7,8]]],
                          [[[1,2],
                            [3,4]],
                           [[5,6],
                            [7,8]],
                           [[9,1],
                            [2,3]]])
        # Do not broadcast matrix axes: (2,1) * (3,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [[1],
                           [2]],
                          [[1,2,3],
                           [4,5,6]])
        # Do not accept less than 2-D arrays: (2) * (2,2)
        self.assertRaises(ValueError,
                          linalg.dot,
                          [1,2],
                          [[1,2,3],
                           [4,5,6]])

class TestBandedSolve(misc.TestCase):

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
        D = 5*np.ones(N, dtype=np.int)

        # Some helpful variables to create the covariances
        W = [np.random.randn(D[i], 2*D[i])
             for i in range(N)]

        # The diagonal blocks (covariances)
        A = [np.dot(W[i], W[i].T) for i in range(N)]
        # The superdiagonal blocks (cross-covariances)
        B = [np.dot(W[i][:,-1:], W[i+1][:,:1].T) for i in range(N-1)]

        C = misc.block_banded(A, B)

        # Create the system to be solved: y=C*x
        x_true = np.random.randn(np.sum(D))
        y = np.dot(C, x_true)
        x_true = np.reshape(x_true, (N, -1))
        y = np.reshape(y, (N, -1))

        #
        # Run tests
        #

        # The correct inverse
        invC = np.linalg.inv(C)

        # Inverse from the function that is tested
        (invA, invB, x, ldet) = linalg.block_banded_solve(np.asarray(A),
                                                          np.asarray(B),
                                                          np.asarray(y))

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
        self.assertTrue(np.allclose(x_true, x))

        # Check the log determinant
        self.assertAlmostEqual(ldet/np.linalg.slogdet(C)[1], 1)

