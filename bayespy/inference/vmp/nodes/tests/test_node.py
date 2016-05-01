################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `dot` module.
"""

import unittest


import numpy as np
import scipy

from numpy import testing

from ..node import Node, Moments

from ...vmp import VB

from bayespy.utils import misc


class TestMoments(unittest.TestCase):

    def test_converter(self):
        """
        Tests complex conversions for moment classes
        """
        
        # Simple one step conversions
        class A(Moments):
            pass
        class B(Moments):
            _converters = {A: lambda x: x+1}
        f = B().get_converter(B)
        self.assertEqual(f(3), 3)
        f = B().get_converter(Moments)
        self.assertEqual(f(3), 3)
        f = B().get_converter(A)
        self.assertEqual(f(3), 4)
        f = A().get_converter(A)
        self.assertEqual(f(3), 3)
        f = A().get_converter(Moments)
        self.assertEqual(f(3), 3)
        self.assertRaises(Moments.NoConverterError,
                          A().get_converter,
                          B)

        # Convert via parent
        class C(B):
            pass
        f = C().get_converter(A)
        self.assertEqual(f(3), 4)

        # Convert via grand parent
        class D(C):
            pass
        class E(D):
            pass
        f = E().get_converter(A)
        self.assertEqual(f(3), 4)

        # Can't convert to child
        self.assertRaises(Moments.NoConverterError,
                          Moments().get_converter,
                          A)
        # Convert to grand child
        class F(Moments):
            _converters = {E: lambda x: 2*x}
        f = F().get_converter(B)
        self.assertEqual(f(3), 6)

        # Use two conversions
        f = F().get_converter(A)
        self.assertEqual(f(3), 2*3+1)
        
        # Can't use child's converter
        class H(Moments):
            pass
        class I(Moments):
            _converters = {A: lambda x: x+1}
        self.assertRaises(Moments.NoConverterError,
                          H().get_converter,
                          A)

        # Conversion to parent is not success
        class J(A):
            pass
        self.assertRaises(Moments.NoConverterError,
                          I().get_converter,
                          J)
        
        # Infinite loop
        class X(Moments):
            pass
        class Y(Moments):
            pass
        X.add_converter(Y, lambda x: x+1)
        Y.add_converter(X, lambda x: x+1)
        self.assertRaises(Moments.NoConverterError,
                          X().get_converter,
                          A)

        # Test that add_converter function does not change the converters of
        # parent classes
        class Z(Moments):
            pass
        class W(Z):
            pass
        W.add_converter(Y, lambda x: x)
        self.assertRaises(Moments.NoConverterError,
                          Z().get_converter,
                          Y)

        # Test that after using add_converter for a child class and then for the
        # parent class, the child class is still able to use the parent's
        # converters
        class X(Moments):
            pass
        class Y(Moments):
            pass
        class A(Moments):
            pass
        class B(A):
            pass
        B.add_converter(Y, lambda x: x+1)
        A.add_converter(X, lambda x: 2*x)
        f = B().get_converter(X)
        self.assertEqual(f(3), 6)

        pass
        
    
class TestNode(misc.TestCase):

    def check_message_to_parent(self, plates_child, plates_message,
                                plates_mask, plates_parent, dims=(2,)):

        # Dummy message
        msg = np.random.randn(*(plates_message+dims))
        # Mask with every other True and every other False
        mask = np.mod(np.arange(np.prod(plates_mask)).reshape(plates_mask),
                      2) == 0

        # Set up the dummy model
        class Dummy(Node):
            _moments = Moments()
            def __init__(self, *args, **kwargs):
                self._parent_moments = len(args)*(Moments(),)
                super().__init__(*args, **kwargs)
            def _get_message_and_mask_to_parent(self, index, u_parent=None):
                return ([msg], mask)
            def _get_id_list(self):
                return []
        parent = Dummy(dims=[dims], plates=plates_parent)
        child = Dummy(parent, dims=[dims], plates=plates_child)

        m = child._message_to_parent(0)[0] * np.ones(plates_parent+dims)

        # Brute-force computation of the message without too much checking
        m_true = msg * misc.squeeze(mask[...,np.newaxis]) * np.ones(plates_child+dims)
        for ind in range(len(plates_child)):
            axis = -ind - 2
            if ind >= len(plates_parent):
                m_true = np.sum(m_true, axis=axis, keepdims=False)
            elif plates_parent[-ind-1] == 1:
                m_true = np.sum(m_true, axis=axis, keepdims=True)

        testing.assert_allclose(m, m_true,
                                err_msg="Incorrect message.")

    def test_message_to_parent(self):
        """
        Test plate handling in _message_to_parent.
        """

        # Test empty plates with scalar messages
        self.check_message_to_parent((),
                                     (),
                                     (),
                                     (),
                                     dims=())
        # Test singular plates
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,3,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,3,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,1,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,3,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,1,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,1,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,3,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,1,4),
                                     (2,1,4))
        # Test missing plates
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (4,3),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (4,3),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (  3,),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (4,3),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (  3,),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (4,3),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (  3,),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (  3,),
                                     (  3,))
        # A complex test
        self.check_message_to_parent((7,6,5,4,3),
                                     (  6,1,4,3),
                                     (1,1,5,4,1),
                                     (  6,5,1,3))
        # Test errors for inconsistent shapes
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (1,3,),
                          (3,),
                          (3,))
        ## self.assertRaises(ValueError, 
        ##                   self.check_message_to_parent,
        ##                   (3,),
        ##                   (3,),
        ##                   (1,3,),
        ##                   (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (1,3,),
                          (1,3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (3,),
                          (4,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (4,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (3,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (3,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (4,),
                          (3,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (3,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (4,),
                          (4,),
                          (1,))


    def test_compute_message(self):
        """
        Test the general sum-multiply function for message computations
        """

        self.assertAllClose(Node._compute_message(3,
                                                  plates_from=(),
                                                  plates_to=(),
                                                  ndim=0),
                            3)

        # Sum over one array
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  plates_from=(3,),
                                                  plates_to=(),
                                                  ndim=0),
                            6)

        # Sum plates
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  [4, 4, 4],
                                                  [5, 5, 5],
                                                  plates_from=(3,),
                                                  plates_to=(),
                                                  ndim=0),
                            20+40+60)

        # Do not sum plates
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  [4, 4, 4],
                                                  [5, 5, 5],
                                                  plates_from=(3,),
                                                  plates_to=(3,),
                                                  ndim=0),
                            [20, 40, 60])

        # Give ndim
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  [4, 4, 4],
                                                  [5, 5, 5],
                                                  plates_from=(),
                                                  plates_to=(),
                                                  ndim=1),
                            [20, 40, 60])

        # Broadcast plates_from
        self.assertAllClose(Node._compute_message(3,
                                                  4,
                                                  5,
                                                  plates_from=(3,),
                                                  plates_to=(),
                                                  ndim=0),
                            3 * (3*4*5))

        # Broadcast plates_to
        self.assertAllClose(Node._compute_message(3,
                                                  4,
                                                  5,
                                                  plates_from=(3,),
                                                  plates_to=(3,),
                                                  ndim=0),
                            3*4*5)

        # Different ndims
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  [4],
                                                  5,
                                                  plates_from=(3,),
                                                  plates_to=(3,),
                                                  ndim=0),
                            [1*4*5, 2*4*5, 3*4*5])

        # Broadcasting dims for some arrays
        self.assertAllClose(Node._compute_message([1, 2, 3],
                                                  [4],
                                                  5,
                                                  plates_from=(),
                                                  plates_to=(),
                                                  ndim=1),
                            [1*4*5, 2*4*5, 3*4*5])

        # Bugfix: Check that plate keys are mapped correctly
        self.assertAllClose(
            Node._compute_message(
                [[1], [2], [3]],
                plates_from=(3,2),
                plates_to=(1,2),
                ndim=0
            ),
            [[6]]
        )

        # Bugfix: Check plate key mapping when plates_to is shorter than shape
        # of the array
        self.assertAllClose(
            Node._compute_message(
                [[1, 2, 3], [4, 5, 6]],
                plates_from=(2,3),
                plates_to=(3,),
                ndim=0
            ),
            [5, 7, 9]
        )

        # Complex example
        x1 = np.random.randn(5,4,1,2,1)
        x2 = np.random.randn(    1,2,1)
        x3 = np.random.randn(5,1,1,1,1)
        self.assertAllClose(Node._compute_message(x1, x2, x3,
                                                  plates_from=(6,5,4,3),
                                                  plates_to=(5,1,1),
                                                  ndim=2),
                            3*6*np.sum(x1*x2*x3, axis=(-4,-3), keepdims=True))

        pass


class TestSlice(misc.TestCase):

    def test_init(self):
        """
        Test the constructor of the X[..] node operator.
        """

        class MyNode(Node):
            _moments = Moments()
            _parent_moments = ()
            def _get_id_list(self):
                return []

        # Integer index
        X = MyNode(plates=(3,4), dims=((),))
        Y = X[2]
        self.assertEqual(Y.plates, (4,))
        X = MyNode(plates=(3,4), dims=((),))
        Y = X[(2,)]
        self.assertEqual(Y.plates, (4,))

        X = MyNode(plates=(3,4), dims=((),))
        Y = X[2,-4]
        self.assertEqual(Y.plates, ())

        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[2,1]
        self.assertEqual(Y.plates, (5,))

        # Full slices
        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[:,1,:]
        self.assertEqual(Y.plates, (3,5,))

        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[1,:,:]
        self.assertEqual(Y.plates, (4,5,))

        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[:,:,1]
        self.assertEqual(Y.plates, (3,4,))

        # Slice with step
        X = MyNode(plates=(9,), dims=((),))
        Y = X[::3]
        self.assertEqual(Y.plates, (3,))

        X = MyNode(plates=(10,), dims=((),))
        Y = X[::3]
        self.assertEqual(Y.plates, (4,))

        X = MyNode(plates=(11,), dims=((),))
        Y = X[::3]
        self.assertEqual(Y.plates, (4,))

        # Slice with a start value
        X = MyNode(plates=(10,), dims=((),))
        Y = X[3:]
        self.assertEqual(Y.plates, (7,))

        # Slice with an end value
        X = MyNode(plates=(10,), dims=((),))
        Y = X[:7]
        self.assertEqual(Y.plates, (7,))

        # Slice with only one element
        X = MyNode(plates=(10,), dims=((),))
        Y = X[6:7]
        self.assertEqual(Y.plates, (1,))

        # Slice starts out of range
        X = MyNode(plates=(10,), dims=((),))
        Y = X[-20:]
        self.assertEqual(Y.plates, (10,))

        # Slice ends out of range
        X = MyNode(plates=(10,), dims=((),))
        Y = X[:20]
        self.assertEqual(Y.plates, (10,))

        # Counter-intuitive: This slice is not empty
        X = MyNode(plates=(3,), dims=((),))
        Y = X[-4::4]
        self.assertEqual(Y.plates, (1,))
        
        # One ellipsis
        X = MyNode(plates=(3,4,5,6), dims=((),))
        Y = X[...,2,1]
        self.assertEqual(Y.plates, (3,4))

        X = MyNode(plates=(3,4,5,6), dims=((),))
        Y = X[2,...,1]
        self.assertEqual(Y.plates, (4,5))

        X = MyNode(plates=(3,4,5,6), dims=((),))
        Y = X[2,1,...]
        self.assertEqual(Y.plates, (5,6))

        # Multiple ellipsis
        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[...,2,...]
        self.assertEqual(Y.plates, (3,5))

        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[...,2,...,...]
        self.assertEqual(Y.plates, (4,5))

        X = MyNode(plates=(3,4,5), dims=((),))
        Y = X[...,...,...,...]
        self.assertEqual(Y.plates, (3,4,5))

        # New axis
        X = MyNode(plates=(3,), dims=((),))
        Y = X[None]
        self.assertEqual(Y.plates, (1,3))
        
        X = MyNode(plates=(3,), dims=((),))
        Y = X[:,None]
        self.assertEqual(Y.plates, (3,1))
        
        X = MyNode(plates=(3,4), dims=((),))
        Y = X[None,:,None,:]
        self.assertEqual(Y.plates, (1,3,1,4))

        #
        # Test errors
        #

        class Z:
            def __getitem__(self, obj):
                return obj

        # Invalid argument
        self.assertRaises(TypeError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()['a'])
        self.assertRaises(TypeError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[[2,1]])

        # Too many indices
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[:,:])
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[...,...,...])

        # Index out of range
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[3])
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[-4])

        # Empty slice
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[3:])
        self.assertRaises(IndexError,
                          MyNode(plates=(3,),
                               dims=((),)).__getitem__,
                          Z()[:-3])
        
        pass

    def test_message_to_child(self):
        """
        Test message to child of X[..] node operator.
        """

        class DummyNode(Node):
            _moments = Moments()
            _parent_moments = (Moments(),)
            def __init__(self, u, **kwargs):
                self.u = u
                super().__init__(**kwargs)
            def _message_to_child(self):
                return self.u
            def _get_id_list(self):
                return []

        # Message not a reference to X.u but a copy of it
        X = DummyNode([np.random.randn(3)],
                      plates=(3,), 
                      dims=((),))
        Y = X[2]
        self.assertTrue(Y._message_to_child() is not X.u,
                        msg="Slice node operator sends a reference to the "
                            "node's moment list as a message instead of a copy "
                            "of the list.")
            
        # Integer indices
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2,1]
        self.assertMessageToChild(Y, [X.u[0][2,1]])

        # Too few integer indices
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2]
        self.assertMessageToChild(Y, [X.u[0][2]])

        # Integer for broadcasted moment
        X = DummyNode([np.random.randn(4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2,1]
        self.assertMessageToChild(Y, [X.u[0][1]])
        X = DummyNode([np.random.randn(4,1)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2,1]
        self.assertMessageToChild(Y, [X.u[0][2,0]])


        # Ignore leading new axes
        X = DummyNode([np.random.randn(3)],
                      plates=(3,), 
                      dims=((),))
        Y = X[None,None,2]
        self.assertMessageToChild(Y, [X.u[0][2]])

        # Ignore new axes before missing+broadcasted plate axes
        X = DummyNode([np.random.randn(3)],
                      plates=(4,3,), 
                      dims=((),))
        Y = X[1,None,None,2]
        self.assertMessageToChild(Y, [X.u[0][2]])

        # New axes
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2,None,None,1]
        self.assertMessageToChild(Y, [X.u[0][2,None,None,1]])

        # New axes for broadcasted axes
        X = DummyNode([np.random.randn(4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[2,1,None,None]
        self.assertMessageToChild(Y, [X.u[0][1,None,None]])

        # Full slice
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[:,2]
        self.assertMessageToChild(Y, [X.u[0][:,2]])

        # Slice with start
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[1:,2]
        self.assertMessageToChild(Y, [X.u[0][1:,2]])

        # Slice with end
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[:2,2]
        self.assertMessageToChild(Y, [X.u[0][:2,2]])

        # Slice with step
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[::2,2]
        self.assertMessageToChild(Y, [X.u[0][::2,2]])

        # Slice for broadcasted axes
        X = DummyNode([np.random.randn(4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[0:2:2,2]
        self.assertMessageToChild(Y, [X.u[0][2]])
        X = DummyNode([np.random.randn(1,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[0:2:2,2]
        self.assertMessageToChild(Y, [X.u[0][0:1,2]])

        # One ellipsis
        X = DummyNode([np.random.randn(3,4)],
                      plates=(3,4), 
                      dims=((),))
        Y = X[...,2]
        self.assertMessageToChild(Y, [X.u[0][...,2]])

        # Ellipsis over broadcasted axes
        X = DummyNode([np.random.randn(5,6)],
                      plates=(3,4,5,6), 
                      dims=((),))
        Y = X[1,...,2]
        self.assertMessageToChild(Y, [X.u[0][:,2]])
        X = DummyNode([np.random.randn(3,1,5,6)],
                      plates=(3,4,5,6), 
                      dims=((),))
        Y = X[1,...,2]
        self.assertMessageToChild(Y, [X.u[0][1,:,:,2]])

        # Multiple ellipsis
        X = DummyNode([np.random.randn(2,3,4,5)],
                      plates=(2,3,4,5), 
                      dims=((),))
        Y = X[...,2,...]
        self.assertMessageToChild(Y, [X.u[0][:,:,2,:]])

        # Ellipsis when dimensions
        X = DummyNode([np.random.randn(2,3,4)],
                      plates=(2,3), 
                      dims=((4,),))
        Y = X[...,2]
        self.assertMessageToChild(Y, [X.u[0][:,2,:]])

        # Indexing for multiple moments
        X = DummyNode([np.random.randn(2,3,4),
                       np.random.randn(2,3)],
                      plates=(2,3), 
                      dims=((4,),()))
        Y = X[1,1]
        self.assertMessageToChild(Y, [X.u[0][1,1],
                                      X.u[1][1,1]])

        pass

    def test_message_to_parent(self):
        """
        Test message to parent of X[..] node operator.
        """

        class ParentNode(Node):
            _moments = Moments()
            _parent_moments = ()
            def _get_id_list(self):
                return []
            
        class ChildNode(Node):
            _moments = Moments()
            _parent_moments = (Moments(),)
            def __init__(self, X, m, mask, **kwargs):
                super().__init__(X, **kwargs)
                self.m = m
                self.mask2 = mask
            def _message_to_parent(self, index, u_parent=None):
                return self.m
            def _mask_to_parent(self, index):
                return self.mask2
            def _get_id_list(self):
                return []

        # General broadcasting
        V = ParentNode(plates=(3,3,3),
                       dims=((),))
        X = V[...]
        m = [ np.random.randn(3,1) ]
        msg = [ np.zeros((1,3,1)) ]
        msg[0][:,:,:] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Integer indices
        V = ParentNode(plates=(3,4),
                 dims=((),))
        X = V[2,1]
        m = [np.random.randn()]
        msg = [ np.zeros((3,4)) ]
        msg[0][2,1] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Integer indices with broadcasting
        V = ParentNode(plates=(3,3),
                 dims=((),))
        X = V[2,2]
        m = [ np.random.randn(1) ]
        msg = [ np.zeros((3,3)) ]
        msg[0][2,2] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Slice indices
        V = ParentNode(plates=(2,3,4,5),
                 dims=((),))
        X = V[:,:2,1:,::2]
        m = [np.random.randn(2,2,3,3)]
        msg = [ np.zeros((2,3,4,5)) ]
        msg[0][:,:2,1:,::2] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Full slice with broadcasting
        V = ParentNode(plates=(2,3),
                 dims=((),))
        X = V[:,:]
        m = [np.random.randn(1)]
        msg = [ np.zeros((1,1)) ]
        msg[0][:] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Start slice with broadcasting
        V = ParentNode(plates=(3,3,3,3),
                 dims=((),))
        X = V[0:,1:,-2:,-3:]
        m = [np.random.randn(1,1)]
        msg = [ np.zeros((1,3,3,1)) ]
        msg[0][:,1:,-2:,:] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # End slice with broadcasting
        V = ParentNode(plates=(3,3,3,3),
                 dims=((),))
        X = V[:2,:3,:4,:-1]
        m = [np.random.randn(1,1)]
        msg = [ np.zeros((3,1,1,3)) ]
        msg[0][:2,:,:,:-1] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Step slice with broadcasting
        V = ParentNode(plates=(3,3,1),
                 dims=((),))
        X = V[::1,::2,::2]
        m = [np.random.randn(1)]
        msg = [ np.zeros((1,3,1)) ]
        msg[0][:,::2,:] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Ellipsis
        V = ParentNode(plates=(3,3,3),
                 dims=((),))
        X = V[...,0]
        m = [np.random.randn(3,3)]
        msg = [ np.zeros((3,3,3)) ]
        msg[0][:,:,0] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)
        
        # New axes
        V = ParentNode(plates=(3,3),
                 dims=((),))
        X = V[None,:,None,None,:]
        m = [np.random.randn(1,3,1,1,3)]
        msg = [ np.zeros((3,3)) ]
        msg[0][:,:] = m[0][0,:,0,0,:]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # New axes with broadcasting
        V = ParentNode(plates=(3,3),
                 dims=((),))
        X = V[None,:,None,:,None]
        m = [np.random.randn(1,3,1)]
        msg = [ np.zeros((1,3)) ]
        msg[0][:,:] = m[0][0,:,0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Multiple messages
        V = ParentNode(plates=(3,),
                 dims=((),()))
        X = V[:]
        m = [np.random.randn(3),
             np.random.randn(3)]
        msg = [ np.zeros((3)), np.zeros((3)) ]
        msg[0][:] = m[0][:]
        msg[1][:] = m[1][:]
        Y = ChildNode(X, m, True, dims=((),()))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Non-scalar variables
        V = ParentNode(plates=(2,3),
                 dims=((4,),))
        X = V[...]
        m = [np.random.randn(2,3,4)]
        msg = [ np.zeros((2,3,4)) ]
        msg[0][:,:,:] = m[0][:,:,:]
        Y = ChildNode(X, m, True, dims=((4,),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Missing values
        V = ParentNode(plates=(3,3,3),
                 dims=((3,),))
        X = V[:,0,::2,None]
        m = [np.random.randn(3,2,1,3)]
        # mask shape: (3, 2, 1)
        mask = np.array([ [[True],  [False]],
                          [[False], [False]],
                          [[False], [True]] ])
        msg = [ np.zeros((3,3,3,3)) ]
        msg[0][:,0,::2,:] = (m[0] * mask[...,None])[:,:,0,:]
        Y = ChildNode(X, m, mask, dims=((3,),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Found bug: int index after slice
        V = ParentNode(plates=(3,3),
                 dims=((),))
        X = V[:,0]
        m = [np.random.randn(3)]
        msg = [ np.zeros((3,3)) ]
        msg[0][:,0] = m[0]
        Y = ChildNode(X, m, True, dims=((),))
        X._update_mask()
        self.assertMessage(X._message_to_parent(0),
                           msg)

        # Found bug: message requires reshaping after reverse indexing
        

        pass
