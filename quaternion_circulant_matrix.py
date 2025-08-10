'''

By 
'''

import numpy as np
import scipy
import unittest

# This is the numpy-quaternion package. Already supports basic quaternion matrix operations.
import quaternion
#import tfquaternion
from circulant import circulant_filter, circulant_filter_2D, vectorize, unvectorize
from quaternion_matrix import quatmatmul_matrix_by_vector
from quaternion_symplectic import qfft, qfft_right, symplectic_decomposition

def quaternionic_circulant_filter(filter):
    """

    :param filter: A 1D or 2D quaternionic signal. 
    :This is the filter kernel. Length M (or MxN)
    :return: A quaternionic circulant matrix. Size MxM (or MNxMN).

    Note: This functions acts as a "constructive" argument for Proposition 3.2, "Quaternionic circulant matrices
    can be written as matrix polynomials"
    """
    if(len(filter.shape) == 1):
        M = len(filter)
        B = quaternion.as_float_array(filter)
        A = np.zeros([M, M, 4], dtype=float)
        for i in range(4):
            A[:, :, i] = circulant_filter(B[:, i])
    elif(len(filter.shape) == 2):
        M, N = filter.shape
        B = quaternion.as_float_array(filter)
        A = np.zeros([M*N, M*N, 4], dtype=float)
        for i in range(4):
            A[:, :, i] = circulant_filter_2D(B[:, :, i])
    else:
        raise ValueError('Input filter must be either 1D or 2D.')
    H = quaternion.as_quat_array(A)
    return H

def create_qft_matrix(N, axis):
    E = np.zeros([N, N], dtype=np.quaternion)
    mu = axis
    wN = np.exp(-mu*2*np.pi/N)
    #print("This is the w_n of the QFT matrix: {}".format(wN))
    for i in range(N):
        for j in range(N):
            E[i, j] = wN ** (i*j)
    E = E / np.sqrt(N)
    return(E)

def quaternion_blur_filter(target_image_shape, sigma2):
    M, N = target_image_shape
    mask_template = np.zeros([M, N], dtype=np.quaternion)
    mask_center = (M//2+1, N//2+1)
    for i in range(M):
        for j in range(N):
            d2 = (i - mask_center[0])**2 + (j - mask_center[1])**2
            if(sigma2 != 0):
                mask_template[i, j] = np.quaternion(np.exp(-d2/(2*sigma2)), 0, 0, 0)
            elif(d2 == 0): # A trivial identity filter
                mask_template[i, j] = np.quaternion(1, 0, 0, 0)              
    mask_template = mask_template / np.sum(mask_template)
    mask_template = np.roll(mask_template, shift=(-mask_center[0],-mask_center[1]), axis=(0,1))
    return(mask_template)

def quaternion_prewitt_filter(target_image_shape, mu, is_horizontal):
    '''
    Implement a Quaternionic Prewitt-mu filter.
    Defined over the spatial domain.
    Discussed in Ell and Sangwine, IEEE TIP 2007 (Sec.VIII/Bi-convolution)

    Returns:
        component_lowpass: This is the low-pass component of the filter, |h|**2 in-paper.
        component_highpass: This is the high-pass component of the filter, h**2 in-paper.

    How to apply:
        Perform left quaternion convolution
    '''
    M, N = target_image_shape
    component_lowpass = np.zeros([M,N],dtype=np.quaternion)
    component_highpass = component_lowpass.copy()
    component_lowpass[0:3,0] = np.quaternion(1, 0, 0, 0) / 6
    component_lowpass[0:3,2] = np.quaternion(1, 0, 0, 0) / 6
    component_highpass[0:3,0] = mu / 6
    component_highpass[0:3,2] = -mu / 6
    if(not is_horizontal):
        component_lowpass = component_lowpass.T
        component_highpass = component_highpass.T
    component_lowpass = np.roll(component_lowpass, shift=(-1,-1), axis=(0,1))
    component_highpass = np.roll(component_highpass, shift=(-1,-1), axis=(0,1))
    return(component_lowpass, component_highpass)

def quaternion_halfprewitt_filter(target_image_shape, mu, theta1, theta2, is_horizontal):
    '''
    Implement a filter that is inspired by the quaternionic Prewitt-like.
    Defined over the spatial domain.

    How to apply:
        Perform left quaternion convolution
    '''
    M, N = target_image_shape
    component_highpass = np.zeros([M,N],dtype=np.quaternion)
    component_highpass[0:3,0] = np.cos(theta1) + mu * np.sin(theta1)
    component_highpass[0:3,2] = np.cos(theta2) + mu * np.sin(theta2)
    component_highpass = component_highpass / 6
    if(not is_horizontal):
        component_highpass = component_highpass.T
    component_highpass = np.roll(component_highpass, shift=(-1,-1), axis=(0,1))
    return(component_highpass)

def quaternion_random_filter(target_image_shape, scaleup_realfactor=1, scaleup_center=5, force_hermitian=False):
    '''
    Implement a random linear 3x3 filter. It is sum-one normalized.
    Defined over the spatial domain.

    How to apply:
        Perform left quaternion convolution
    '''
    M, N = target_image_shape
    res = np.zeros([M, N, 4])
    res[0:3, 0:3, :] = np.random.rand(3, 3, 4)
    res[:, :, 0] = res[:, :, 0] * scaleup_realfactor #Scale up the real part
    if(force_hermitian):
        res[0:3, 0:3, 0] = .5*(res[0:3, 0:3, 0] + res[0:3, 0:3, 0].T)
        res[0:3, 0:3, 1] = .5*(res[0:3, 0:3, 1] - res[0:3, 0:3, 1].T)
        res[0:3, 0:3, 2] = .5*(res[0:3, 0:3, 2] - res[0:3, 0:3, 2].T)
        res[0:3, 0:3, 3] = .5*(res[0:3, 0:3, 3] - res[0:3, 0:3, 3].T)
    res[1, 1, 0] += scaleup_center
    res = quaternion.as_quat_array(res)
    res = res / np.sum(res)
    res = np.roll(res, shift=(-1,-1), axis=(0,1))
    return(res)


#######################

def quaternion_circulant_by_vector_product(C=None, Ckernel=None, 
    Ceigenvalues_forward=None,  # This is FT^mu_R{k_C}            ie the forward transform with axis mu
    Ceigenvalues_backward=None, # This is FT^{-mu}_R{k_C}         ie the forward transform with axis -mu 
    v=None,                     #(or the backward transfrom with axis +mu, but fais gaffe at the normalization constant..) 
    mu=None):
    """
    Computes the product of a quaternionic circulant matrix C by vector v.
    See circulant.circulant_by_vector_product for details.
    """
    if(v is None):
        raise ValueError("Vector v was not provided.")
    if(C is not None):
        assert(type(C) == np.ndarray and type(C.flat[0]) == np.quaternion)
        return(quatmatmul_matrix_by_vector(C, v))
    if(Ckernel is not None):
        # We need the *right* QFT; the left QFT is more connected to C^H.
        # Note: I multiply qfft_right by np.sqrt(N) because the argument is 'eigenvalues'.
        assert(mu is not None)
        N = len(Ckernel)
        return(quaternion_circulant_by_vector_product(C=None, Ckernel=None, 
                Ceigenvalues_forward = np.sqrt(N) * qfft_right(Ckernel, qft_axis=mu, apply_shift=False),
                Ceigenvalues_backward = np.sqrt(N) * qfft_right(Ckernel, qft_axis=-mu, apply_shift=False),
                v=v, mu=mu))
    if(Ceigenvalues_forward is not None):
        assert(Ceigenvalues_backward is not None)
        assert(mu is not None)
        N = len(Ceigenvalues_forward)
        assert(N == len(Ceigenvalues_backward))
        H_mu = Ceigenvalues_forward / np.sqrt(N)
        H_minusmu = Ceigenvalues_backward / np.sqrt(N)
        F = qfft_right(v, qft_axis=mu, apply_shift=False)
        (F_simplex, F_perplex), (_, mu2, _) = symplectic_decomposition(F, mu1=mu, save_true_values=True)
        res = np.sqrt(N) * qfft_right(H_mu*F_simplex + H_minusmu*F_perplex*mu2, qft_axis=-mu, apply_shift=False)
        return(res)

def quaternion_doublyblockCirculant_by_vector_product(C=None, Ckernel=None, 
    Ceigenvalues_forward=None,
    Ceigenvalues_backward=None, 
    v=None, mu=None, M=None):
    """
    Computes the product of a doubly-block *quaternion* circulant matrix C by vector v.
    See circulant.doublyblockCirculant_by_vector_product for details.
    """
    if(v is None):
        raise ValueError("Vector v was not provided.")
    if(C is not None):
        return(C @ v)
    if(Ckernel is not None):
        assert(M is not None)
        assert(mu is not None)
        N = len(Ckernel) // M
        Ckernel_unvectorized = unvectorize(Ckernel, M, N)
        #M2, N2 = Ckernel.shape
        #if(M is not None):
        #    assert(M == M2)
        return(quaternion_doublyblockCirculant_by_vector_product(C=None, Ckernel=None, 
                Ceigenvalues_forward=np.sqrt(M*N)*vectorize(qfft_right(Ckernel_unvectorized, qft_axis=mu, apply_shift=False)), 
                Ceigenvalues_backward=np.sqrt(M*N)*vectorize(qfft_right(Ckernel_unvectorized, qft_axis=-mu, apply_shift=False)), 
                v=v, mu=mu, M=M
                )
            )
    if(Ceigenvalues_forward is not None):
        assert(Ceigenvalues_backward is not None)
        assert(len(Ceigenvalues_forward) == len(Ceigenvalues_backward))
        assert(mu is not None)
        assert(M is not None)
        #if(len(Ceigenvalues_forward.shape) != 2 and len(Ceigenvalues_backward.shape) != 2):
        #    raise ValueError('Please provide a 2D matrix for Ceigenvalues_forward and backward.')
        N = len(Ceigenvalues_forward) // M
        H_mu = unvectorize(Ceigenvalues_forward, M, N) / np.sqrt(M*N)
        H_minusmu = unvectorize(Ceigenvalues_backward, M, N) / np.sqrt(M*N)
        F = qfft_right(unvectorize(v, M, N), qft_axis=mu, apply_shift=False)
        (F_simplex, F_perplex), (_, mu2, _) = symplectic_decomposition(F, mu1=mu, save_true_values=True)
        res = np.sqrt(M*N) * qfft_right(H_mu*F_simplex + H_minusmu*F_perplex*mu2, qft_axis=-mu, apply_shift=False)
        return(vectorize(res))


class TestQCirculant(unittest.TestCase):
    def test_filter1D(self):
        a = np.quaternion(+1, 0, 0, 0)
        b = np.quaternion(-1, 0, 0, 0)
        c = np.quaternion( 0, 0, 0, 0)
        filter = np.array(
            [a, b, c],
        )
        out = circulant_filter(filter)
        desired = np.array(
            [
                [a, c, b],
                [b, a, c],
                [c, b, a],
            ]
        )
        self.assertTrue((out == desired).all())

    def test_hermitian_circulant(self):
        # Check our function to check for hermitian-ity of a quaternion circulant
        pass

class TestQFT_generic(unittest.TestCase):
    def test_qft_and_dft(self):
        # Properties of the QFT matrix:
        # This tests that Q_N^i = A_N 
        N = 4
        E = create_qft_matrix(N, np.quaternion(0, 1, 0, 0))
        A = scipy.linalg.dft(N, scale='sqrtn')
        AX = np.zeros([N, N, 4])
        AX[:, :, 0] = np.real(A)
        AX[:, :, 1] = np.imag(A)
        QAX = quaternion.as_quat_array(AX)
        self.assertTrue((E == QAX).all())

    def test_qft_and_dft_2(self):
        # Properties of the QFT matrix:
        # This tests that Q_N^i = A_N (and not Q_N^j = A_N)
        N = 4
        E = create_qft_matrix(N, np.quaternion(0, 0, 1, 0))
        A = scipy.linalg.dft(N, scale='sqrtn')
        AX = np.zeros([N, N, 4])
        AX[:, :, 0] = np.real(A)
        AX[:, :, 1] = np.imag(A)
        QAX = quaternion.as_quat_array(AX)
        self.assertFalse((E == QAX).all())

if __name__=='__main__':
    unittest.main()