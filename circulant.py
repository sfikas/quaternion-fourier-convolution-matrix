'''
Routines on circulant and doubly-block circulant matrices
G.Sfikas 2017-2020
'''

import numpy as np
from PIL import Image
from scipy.fftpack import fft, fft2, ifft, ifft2
import unittest

class TestVectorize(unittest.TestCase):
    def test_vectorize_1(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        v = vectorize(A, stack_columns=True, flip_row_order=False)
        self.assertTrue( (v == [1, 4, 2, 5, 3, 6]).all() )

    def test_vectorize_15(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        v = vectorize(A, stack_columns=False, flip_row_order=False)
        self.assertTrue( (v == [1, 2, 3, 4, 5, 6]).all() )

    def test_vectorize_2(self):
        M = 80
        N = 120
        v = np.random.randn(M, N)
        vecd = vectorize(v)
        vv = unvectorize(vecd, M, N)
        self.assertTrue((v == vv).all())

class TestCirculant(unittest.TestCase):
    def test_filter1D(self):
        filter = np.array([1, -1, 0])
        out = circulant_filter(filter)
        desired = np.array(
            [
                [1, 0, -1],
                [-1, 1, 0],
                [0, -1, 1],
            ]
        )
        self.assertTrue((out == desired).all())
    def test_filter1D_2(self):
        filter = np.array([1, 0, 0])
        out = circulant_filter(filter)
        desired = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        self.assertTrue((out == desired).all())
    def test_filter1D_3(self):
        filter = np.array([0, 0, 0])
        out = circulant_filter(filter)
        desired = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        self.assertTrue((out == desired).all())

class TestDoublyBlockCirculant(unittest.TestCase):
    def test_filter(self):
        r = circulant_filter_2D(np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, -1, 0],
            ]
        ))
        desired = np.array([
            [ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.],
            [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.],
            [ 0.,  1.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0., -1.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0., -1.],
            [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  1.],
        ])

class TestProducts(unittest.TestCase):
    def test_1d_product(self):
        filter = np.array([1, 2, 3])
        C = circulant_filter(filter)
        v = np.array([-3, 2, 6])
        #
        result_1 = circulant_by_vector_product(C=C, Ckernel=None, Ceigenvalues=None, v=v)
        result_2 = circulant_by_vector_product(C=None, Ckernel=filter, Ceigenvalues=None, v=v)
        for i in range(len(filter)):
            self.assertAlmostEqual(result_1[i], result_2[i])
        result_3 = circulant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=fft(filter), v=v)
        for i in range(len(filter)):
            self.assertAlmostEqual(result_2[i], result_3[i])

    def test_2d_product(self):
        M = 3
        N = 2
        filter = pad_to_size(np.array(
            [
                [2, -1],
                [0,  1],
            ]
        ), M, N)
        C = circulant_filter_2D(pad_to_size(filter, M, N)) #TODO: second pad_to_size, check and correct this
        v = np.array([1, 2, 3, 4, -1, -2])
        result_1 = doublyblockCirculant_by_vector_product(C=C, Ckernel=None, Ceigenvalues=None, v=v, M=None)
        result_2 = doublyblockCirculant_by_vector_product(C=None, Ckernel=filter, Ceigenvalues=None, v=v, M=M)
        for i in range(M*N):
            self.assertAlmostEqual(result_1[i], result_2[i])
        result_3 = doublyblockCirculant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=vectorize(fft2(filter)), v=v, M=M)
        for i in range(M*N):
            self.assertAlmostEqual(result_2[i], result_3[i])

    def test_2d_product_2(self):
        M = 30
        N = 20
        filter = pad_to_size(np.array(
            [
                [2, -1],
                [0,  1],
            ]
        ), M, N)
        C = circulant_filter_2D(pad_to_size(filter, M, N))
        v = np.random.randn(M*N)
        result_1 = doublyblockCirculant_by_vector_product(C=C, Ckernel=None, Ceigenvalues=None, v=v, M=None)
        result_2 = doublyblockCirculant_by_vector_product(C=None, Ckernel=filter, Ceigenvalues=None, v=v, M=M)
        for i in range(M*N):
            self.assertAlmostEqual(result_1[i], result_2[i])
        result_3 = doublyblockCirculant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=vectorize(fft2(filter)), v=v, M=M)
        for i in range(M*N):
            self.assertAlmostEqual(result_2[i], result_3[i])

###################################################################################################################

def circulant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=None, v=None):
    """
    Computes the product of a circulant matrix C by vector v.
    The matrix C must be given as either:
    * The actual NxN matrix  ("C")
    * The kernel of the corresponding circular convolution, sized Nx1 ("Ckernel")
    * The eigenvalues of C ("Ceigenvalues"), sized Nx1
    """
    if(v is None):
        raise ValueError("Vector v was not provided.")
    if(C is not None):
        return(C @ v)
    if(Ckernel is not None):
        return(circulant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=fft(Ckernel), v=v))
    if(Ceigenvalues is not None):
        return(ifft(Ceigenvalues * fft(v)))

def doublyblockCirculant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=None, v=None, M=None):
    """
    Computes the product of a doubly-block circulant matrix C by vector v.
    The first step of this process is to unvectorize "v", sized MN, to an image of size MxN. 
    "M", ie the number of rows of the unvectorized image, should be user-provided.
    The matrix C must be given as either:
    * The actual MNxMN matrix  ("C")
    * The kernel of the corresponding circular convolution, sized MxN ("Ckernel")
    * The eigenvalues of C ("Ceigenvalues"), sized MNx1
    """
    if(v is None):
        raise ValueError("Vector v was not provided.")
    if(C is not None):
        return(C @ v)
    if(Ckernel is not None):
        if(M is None):
            raise ValueError("The user should provide a value for M.")
        return(doublyblockCirculant_by_vector_product(C=None, Ckernel=None, Ceigenvalues=vectorize(fft2(Ckernel)), v=v, M=M))
    if(Ceigenvalues is not None):
        if(M is None):
            raise ValueError("The user should provided a value for M.")
        h_2D = unvectorize(Ceigenvalues, M, len(Ceigenvalues) // M)
        v_2D = unvectorize(v, M, len(v) // M)
        return(vectorize( np.real(ifft2(h_2D * fft2(v_2D))))
            )

###################################################################################################################

def circulant_filter(filter):
    """

    :param filter: A 1D signal. This is the filter kernel. Length M.
    :return: A circulant matrix. Size MxM .
    """
    M = filter.shape[0]
    res = np.zeros([M, M])
    for i in range(M):
        for j in range(M):
            res[i, j] = filter[(i-j)%M]
    return res

def circulant_filter_2D(filter):
    """

    :param filter: The MxN filter.
    :return: A doubly block circulant matrix. Each of the M blocks is NxN. The output is therefore MNxMN.
    """
    input_size = filter.shape
    M = input_size[0]
    N = input_size[1]
    blocks = []

    for i in range(M):
        blocks.append(circulant_filter(filter[-i, :]))  #was -i-1
    res = np.zeros([M*N, M*N])
    for i in range(M):
        for j in range(M):
            current_block = blocks[(i-j)%M]
            start_i = i*N
            start_j = j*N
            end_i = (i+1)*N
            end_j = (j+1)*N
            res[start_i:end_i, start_j:end_j] = current_block
    return res

def vectorize(im, stack_columns=False, flip_row_order=True):
    """

    :param im: Input gray-scale image, MxN.
    :return: Vectorized image, size MNx1.
    """
    if(flip_row_order):
        im = np.flipud(im)
    if(stack_columns):
        res = im.flatten(order='F')
    else:
        res = im.flatten(order='C')
    return res

def unvectorize(vec, M, N, assume_stacked_columns=False, flip_row_order=True):
    """

    :param vec: Vectorized form of image, MNx1.
    :return: Image size MxN.
    """
    if(assume_stacked_columns):
        res = np.reshape(vec, [M, N], order='F')
    else:
        res = np.reshape(vec, [M, N], order='C')
    if(flip_row_order):
        res = np.flipud(res)
    return res

def pad_to_size(filter, M, N, shift=0):
    input_size = filter.shape
    Morig = input_size[0]
    Norig = input_size[1]
    res = np.zeros([M, N])
    res = np.zeros([M, N])
    res[:Morig, :Norig] = filter
    if(shift != 0):
        tt = res[:, 0:shift].copy()
        res[:, :-shift] = res[:, shift:]
        res[:, -shift:] = tt
        tt = res[0:shift, :].copy()
        res[:-shift, :] = res[shift:, :]
        res[-shift:, :] = tt
    return res #The above is not necessary

if __name__=='__main__':
    unittest.main()