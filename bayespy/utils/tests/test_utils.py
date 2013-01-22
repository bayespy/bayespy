
import unittest

import numpy as np

from bayespy.utils import utils


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

if __name__ == '__main__':
    unittest.main()
