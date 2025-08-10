'''

By XXXXXXXX, Fall 2019
'''

import logging
from operator import is_
import numpy as np
import tqdm
import unittest
import itertools

# This is the numpy-quaternion package. Already supports basic quaternion matrix operations.
import quaternion
#import tfquaternion
from scipy.linalg import diagsvd

def isHermitian(A, rtol=1e-05, atol=1e-08):
    #https://stackoverflow.com/a/52601850/5615276
    #return np.allclose(A, A.T, rtol=rtol, atol=atol)
    return np.all(np.abs(A-A.conj().T) < rtol)

def quatlength(q):
    '''
    Computes the length of a single quaternion
    '''
    assert(type(q) == np.quaternion)
    # is q.w the same as q.real ? (the docs say q.w is the scalar part)
    return(np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2))

def quatmatmul_matrix_by_matrix(K, L, fast_version=True):
    # Code for test (TODO)
    #A = quaternion.as_quat_array(np.random.randn(64, 128, 4))
    #B = quaternion.as_quat_array(np.random.randn(128, 64, 4))
    #tt1 = quatmatmul_matrix_by_matrix(A,B, fast_version=True)
    #tt2 = quatmatmul_matrix_by_matrix(A,B, fast_version=False
    #print(np.sum(tt1 - tt2))
    assert(type(K) == np.ndarray)
    assert(type(L) == np.ndarray)
    assert(len(K.shape) == 2)
    assert(len(L.shape) == 2)
    assert(type(K[0,0]) == np.quaternion)
    assert(type(L[0,0]) == np.quaternion)
    M, N = K.shape
    Ma, Na = L.shape
    assert(Ma == N)
    if not fast_version:
        res = np.zeros(shape=[M,Na], dtype=np.quaternion)
        for i in range(M):
            for j in range(Na):
                res[i,j] = quatdotproduct(K[i,:], L[:, j], dont_take_conjugate=True)
        return(res)
    else:
        #
        # K = K0 + K1*i + K2*j + K3*k
        # L = L0 + L1*i + L2*j + L3*k
        #
        # KL:
        # Real part = K0*L0 -K1*L1 -K2*L2 -K3*L3
        # i         = K0*L1 +K1*L0 +K2*L3 -K3*L2
        # j         = K0*L2 -K1*L3 +K2*L0 +K3*L1
        # k         = K0*L3 +K1*L2 -K2*L1 +K3*L0
        Kfloat = quaternion.as_float_array(K)
        Lfloat = quaternion.as_float_array(L)
        res_4x4 = np.zeros([M, Na, 4, 4])
        for i in range(4):
            for j in range(4):
                res_4x4[:, :, i, j] = Kfloat[:, :, i] @ Lfloat[:, :, j]

        res = np.zeros([M, Na, 4])
        res[:, :, 0] = res_4x4[:, :, 0, 0] - res_4x4[:, :, 1, 1] - res_4x4[:, :, 2, 2] - res_4x4[:, :, 3, 3]
        res[:, :, 1] = res_4x4[:, :, 0, 1] + res_4x4[:, :, 1, 0] + res_4x4[:, :, 2, 3] - res_4x4[:, :, 3, 2]
        res[:, :, 2] = res_4x4[:, :, 0, 2] - res_4x4[:, :, 1, 3] + res_4x4[:, :, 2, 0] + res_4x4[:, :, 3, 1]
        res[:, :, 3] = res_4x4[:, :, 0, 3] + res_4x4[:, :, 1, 2] - res_4x4[:, :, 2, 1] + res_4x4[:, :, 3, 0]
        return(quaternion.as_quat_array(res))

def quatdotproduct(x, y, dont_take_conjugate=False):
    '''
    Computes a dot product between quaternionic vectors
    '''
    assert(type(x) == np.ndarray)
    assert(type(y) == np.ndarray)
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    N = x.shape[0]
    res = quaternion.quaternion(0, 0, 0, 0)
    if(not dont_take_conjugate):
        for i in range(N):
            res += (x[i].conj())*y[i]
    else:
        for i in range(N):
            res += x[i]*y[i]
    return(res)


def quatmatmul_matrix_by_vector(A, x, multiplyfromright=True):
    '''
    Computes the product of a quaternionic matrix by a quaternionic vector, i.e. Ax .
    The result is a quaternionic vector.
    TODO: Do a fast version as in quatmatrix by matrix.
    '''
    assert(type(A) == np.ndarray)
    assert(type(x) == np.ndarray)
    assert(len(A.shape) == 2)
    assert(len(x.shape) == 1)
    M = A.shape[0]
    N = A.shape[1]
    assert(N == x.shape[0])
    res = np.zeros([M,], dtype=np.quaternion)
    for i in range(N):
        if multiplyfromright:
            res += A[:, i]*x[i] #This is a *right* scalar multiple of columns
        else:
            res += x[i]*A[:, i]
    return(res)

def quatmatmul_vector_by_matrix(x, A):
    '''
    Computes the product of a quaternionic vector by a quaternionic matrix, i.e. xA .
    The result is a quaternionic vector.
    '''
    assert(type(A) == np.ndarray)
    assert(type(x) == np.ndarray)
    assert(len(A.shape) == 2)
    assert(len(x.shape) == 1)
    M = A.shape[0]
    N = A.shape[1]
    raise NotImplementedError('TODO: Check that it works ok for non-square matrices.')
    assert(M == x.shape[0])
    res = np.zeros([N,], dtype=np.quaternion)
    for i in range(M):
            res += x[i]*A[i, :] #This is a *left* scalar multiple of rows
    return(res)

def chi_inverse(C, strict_mode=True):
    '''
    Accepts as input the complex adjoint 2Mx2N complex matrix and outputs the 'original' MxN quaternion matrix.
    '''
    assert(type(C) == np.ndarray)
    M2 = C.shape[0]
    N2 = C.shape[1]
    assert(M2 % 2 == 0)
    assert(N2 % 2 == 0)
    M = M2 // 2
    N = N2 // 2
    Q = np.zeros([M, N], dtype=np.quaternion)
    # Check that the input is indeed complex adjoint
    A1 = C[:M, :N]
    A2 = C[:M, N:]
    A2mc = C[M:, :N]
    A1c = C[M:, N:]
    if(strict_mode):
        np.testing.assert_array_almost_equal(A1, np.conj(A1c))
        np.testing.assert_array_almost_equal(A2, -np.conj(A2mc))
    else:
        # Make sure that the blocks of the input are *made* to "match"
        #A1 = .5*(A1 + np.conj(A1c))
        #A2 = .5*(A2 - np.conj(A2mc))
        pass
    # Create output quaternion matrix
    # Assumes a Cayley-Dickson expansion: A = A1+A2j.
    A = np.real(A1)*quaternion.one + np.imag(A1)*quaternion.x + np.real(A2)*quaternion.y + np.imag(A2)*quaternion.z
    return(A)

def chi(Q):
    '''
    Accepts as input a Quaternion matrix "Q", size MxN, and outputs its complex adjoint matrix, size 2Mx2N.
    '''
    assert(type(Q) == np.ndarray)
    assert(len(Q.shape) == 2)
    assert(type(Q[0, 0]) == np.quaternion)
    M = Q.shape[0]
    N = Q.shape[1]
    C = np.zeros([2*M, 2*N], dtype=np.complex)
    # Decompose input into MxNx4 real tensor
    tt = quaternion.as_float_array(Q)
    A1 = tt[:, :, 0] + 1j*tt[:, :, 1]
    A2 = tt[:, :, 2] + 1j*tt[:, :, 3]
    C[:M, :N] = A1
    C[:M, N:] = A2
    C[M:, :N] = np.conj(-A2)
    C[M:, N:] = np.conj(A1)
    return(C)

def quaternion_matrix_inverse(A):
    from scipy.linalg import inv
    '''
    Compute inverse of quaternion matrix A.
    This uses the following relation (inv represents multiplicative inverse)
    chi(inv(A)) = inv(chi(A)) [Zhang97 Theorem 4.2, Lee49],
    therefore..
    inv(A) = chi_inverse(inv(chi(A)))
    '''
    tt = inv(chi(A))
    res = chi_inverse(tt, strict_mode=False)
    return(res)

def eigdecomposition_hermitian(Q):
    return(eigdecomposition(Q, is_hermitian=True))

def eigdecomposition(Q, is_hermitian=True, return_nparrays=False):
    '''
    Computes the eigendecomposition of a Quaternionic input Q.
    Outputs:
        * A list of (real) eigenvalues, sorted.
        * A list of corresponding quaternionic eigenvectors.
    For example:
        L, V = eigdecomposition_hermitian(Q)
        To get, say, the 3rd eigenvalue and eigenvector, use L[2] and V[2] respectively.
        (this is somewhat different that numpy.linalg.eigh)
    '''
    assert(type(Q) == np.ndarray)
    assert(len(Q.shape) == 2)
    assert(type(Q[0, 0]) == np.quaternion)
    assert(Q.shape[0] == Q.shape[1]) #Must be rectangular
    #np.testing.assert_array_almost_equal(Q, Q.conj().T) #Must be Hermitian
    #
    N = Q.shape[0]
    if is_hermitian:
        lam, V = np.linalg.eigh(chi(Q))
    else:
        lam, V = np.linalg.eig(chi(Q))
    if(not return_nparrays):
        idiodianysmata = []
        idiotimes = []
    else:
        idiodianysmata = np.zeros([N,N], dtype=np.quaternion)
        idiotimes = np.zeros([N,], dtype=np.quaternion)
    for i in range(N):
        # TODO WARNING !!!! Check this (see qsvd), possible problem with index
        i2 = 2*i-1
        if(not return_nparrays):
            idiotimes.append(lam[i2])
        else:
            idiotimes[i] = lam[i2]
        if(i == 0):
            current_eigenvalue = lam[i2]
        else:
            #TODO: Fix this, this is obviously wrong
            previous_eigenvalue = current_eigenvalue
            if(previous_eigenvalue < current_eigenvalue):
                raise ValueError('Eigenvalues of eig/eigh are not sorted - they should have been! (Add a sort command in-code to solve this)')
            current_eigenvalue = previous_eigenvalue
        tt = V[:, i2]
        ttA = tt[:N]
        ttB = tt[N:]
        ttB = -np.conj(ttB)
        ttA1 = np.real(ttA)
        ttA2 = np.imag(ttA)
        ttB1 = np.real(ttB)
        ttB2 = np.imag(ttB)
        neo_idiodianysma = ttA1*quaternion.one + ttA2*quaternion.x + ttB1*quaternion.y + ttB2*quaternion.z
        if(not return_nparrays):
            idiodianysmata.append(neo_idiodianysma)
        else:
            idiodianysmata[:, i] = neo_idiodianysma
        # Check that eigenvector is indeed normalized (remove for better speed..)
        #print(neo_idiodianysma[0]*neo_idiodianysma[0].conj() + neo_idiodianysma[1]*neo_idiodianysma[1].conj())
    #print(idiotimes, idiodianysmata)
    #exit(1)
    return(idiotimes, idiodianysmata)



def bidiagonalization(A):
    def householder_vector(x, v):
        # x: A quaternionic vector (In paper this is 'a', the input vector)
        # v: A real vector, of same size as x
        #
        # Returns:
        #   u:      quaternion vector, same size as inputs
        #   zeta:   quaternion scalar
        def qnorm(v):
            v = np.atleast_1d(v)
            res = np.sqrt(quatdotproduct(v, v))
            return(res.real)

        def dotproduct(x, y, dont_take_conjugate=False):
            N = x.shape[0]
            res = quaternion.quaternion(0, 0, 0, 0)
            for i in range(N):
                res += x[i]*y[i]
            return(res)
        x = np.atleast_1d(np.squeeze(x))
        v = np.atleast_1d(np.squeeze(v))
        alpha = qnorm(x)
        if(alpha == 0):
            u = x - x # This ensures that u is zero, even if a is a complex quaternion vector with zero norm.
            zeta = np.quaternion(1, 0, 0, 0)
            return(u, zeta)
        dotproduct = quatdotproduct(v, x, dont_take_conjugate=True)
        r = np.abs(dotproduct)
        if r != 0:
            zeta = -dotproduct / r
        else:
            zeta = np.quaternion(1, 0, 0, 0)
        mu = np.sqrt(alpha * (alpha + r))
        u = (x - (zeta*v)*alpha) / mu
        return u, zeta

    def mytransform_left(T, h):
        #T - h * (T' * h)'
        # T: square quaternion matrix
        # h:  quaternion vector
        h = np.atleast_1d(h)
        t1 = quatmatmul_matrix_by_vector(T.conj().T, h)
        t2 = np.outer(h, t1.conj())
        return(T - t2)

    def mytransform_right(T, h):
        #T - (T * h.') * conj(h)
        # T: quaternion matrix
        # h: quaternion vector
        #print(T.shape, h.shape)
        h = np.atleast_1d(h)
        h = h.conj()
        t1 = quatmatmul_matrix_by_vector(T, h)
        t2 = np.outer(t1, h.conj())
        return(T - t2)

    def quateye(n):
        res = np.zeros((n, n), dtype=np.quaternion)
        for i in range(n):
            res[i,i] = np.quaternion(1, 0, 0, 0)
        return(res)

    r, c = A.shape
    if(c > r):
        V, B, U = bidiagonalization(A.conj().T)
        return(U.conj().T, B.T, V.conj().T)

    assert(c <= r)
    # Iterate over the whole matrix, dealing with one column and one row on
    # each pass through the loop.
    #U = np.eye(r, dtype=A.dtype)  # Initialize the three matrix results
    U = quateye(r)
    B = A.copy()                  # to appropriately sized matrices.
    V = quateye(c)
    for i in range(c):
        # Compute and apply a left Householder transformation to the ith
        # column (part of the column for the second and subsequent columns).
        h, zeta = householder_vector(B[i:, i], np.eye(r - i, 1))
        #tt = T - h * (T' * h)'
        B[i:, i:] = (1. / zeta) * mytransform_left(B[i:, i:], h)
        U[i:, :]  = (1. / zeta) * mytransform_left(U[i:, :], h)
        
        if i == c - 1:
            return U, B, V  # On the last column, we are done.        
        else:        
            # Now compute and apply a right Householder transformation to the ith
            # row. In the case of the first row, this excludes the first element,
            # and in later rows, the first i elements.        
            #B(i:end, j:end) = (T - (T * h.') * g) ./zeta;
            #B[i:, j:] = (T - np.dot(T, h)[:, np.newaxis] * g) / zeta  CHATGPT IDEA
            #V( :   , j:end) = (T - (T * h.') * g) ./zeta;                
            #V[:, j:] = (T - np.dot(T, h)[:, np.newaxis] * g) / zeta    CHATGPT IDEA
            j = i + 1
            h, zeta = householder_vector(B[i, j:], np.eye(1, c - i - 1))
            B[i:, j:] = mytransform_right(B[i:, j:], h) * (1. / zeta)        
            V[:,  j:] = mytransform_right(V[:, j:], h) * (1. / zeta)        



def complex2quaternionmatrix(A):
    assert(type(A) == np.ndarray)
    res = np.zeros([A.shape[0], A.shape[1], 4])
    realpart = np.real(A)
    imagpart = np.imag(A)
    res[..., 0] = realpart
    res[..., 1] = imagpart
    return(quaternion.as_quat_array(res))

def qsvd(A, return_also_bidiagonalization=False):
    #
    # Returns U, S, VH,A = U @ S @ V.H
    #
    # if flag is set:
    #   Returns LH, Uaux, Sigma, VauxH, RH,
    #   where A = LH @ Uaux @ Sigma @ VauxH @ RH
    #
    L, B, R = bidiagonalization(A)  # This means that L.H B R.H = A
    Uaux, Sigma, VauxH = np.linalg.svd(quaternion.as_float_array(B)[:, :, 0])
    if not return_also_bidiagonalization:
        U = quatmatmul_matrix_by_matrix(L.conj().T, complex2quaternionmatrix(Uaux))
        V = quatmatmul_matrix_by_matrix(complex2quaternionmatrix(VauxH), R.conj().T)
        return(U, Sigma, V)
    else:
        return(L.conj().T, Uaux, Sigma, VauxH, R.conj().T)

def qsvd_synthesis(A, B, C):
    Sigma = np.zeros([A.shape[0], C.shape[0]], dtype=np.quaternion)
    Sigma = quaternion.as_float_array(Sigma)
    Sigma[:, :, 0] = diagsvd(B, A.shape[0], C.shape[0])
    Sigma = quaternion.as_quat_array(Sigma)
    tt = quatmatmul_matrix_by_matrix(A, Sigma)
    return(quatmatmul_matrix_by_matrix(tt, C))
    '''
    Sigma = np.zeros_like(A, dtype=np.quaternion)
    for i in range(A.shape[1]):
        Sigma[i, i] = B[i]
    tt = quatmatmul_matrix_by_matrix(A, Sigma)
    return(quatmatmul_matrix_by_matrix(tt, C))
    '''

def qsvd_chi(Q):
    '''
    Computes the SVD of a Quaternionic input Q.
    u, s, vh = qsvd(Q)
    '''
    raise KeyError('Don"t use this version, use qsvd')
    def convert2quaternion(tt):
        ttA = tt[:N]
        ttB = tt[N:]
        ttB = -np.conj(ttB)
        ttA1 = np.real(ttA)
        ttA2 = np.imag(ttA)
        ttB1 = np.real(ttB)
        ttB2 = np.imag(ttB)
        return(ttA1*quaternion.one + ttA2*quaternion.x + ttB1*quaternion.y + ttB2*quaternion.z)
    assert(type(Q) == np.ndarray)
    assert(len(Q.shape) == 2)
    assert(type(Q[0, 0]) == np.quaternion)
    N = Q.shape[0]
    aristera = np.zeros([N,N], dtype=np.quaternion)
    idiazouses = np.zeros([N,], dtype=np.quaternion)
    dexia_h = np.zeros([N,N], dtype=np.quaternion)
    
    u, lam, vh = np.linalg.svd(chi(Q))
    for i in range(N):
        #i2 = 2*i-1
        i2 = 2*i
        idiazouses[i] = lam[i2]
        aristera[:, i] = convert2quaternion(u[:, i2])
        dexia_h[i, :] = convert2quaternion(vh[i2, :])
    return(aristera, idiazouses, dexia_h)


class TestBasic(unittest.TestCase):
    def test_selftest(self):
        # This 'tests' if Euler's identity for quaternions works.
        mu = np.quaternion(0, 10, -10, 4).normalized()
        theta = np.random.rand()*2*np.pi
        leftside =  np.exp(mu*theta)
        rightside = np.cos(theta) + mu*np.sin(theta)
        self.assertAlmostEqual( leftside, rightside )

    def test_createQuatMatrix(self):
        # Create a Quaternion matrix
        A = np.zeros([2, 2], dtype=np.quaternion)

    def test_quaternion_hadamard_product(self):
        M = 15
        N = 20
        A = np.zeros([M, N], dtype=np.quaternion)
        B = np.zeros_like(A)
        for i in range(M):
            for j in range(N):
                A[i,j] = np.quaternion(1, 5, i, j)
                B[i,j] = np.quaternion(i, j, i*j, -i+5)
        C = A*B 
        for i in range(M):
            for j in range(N):
                self.assertAlmostEqual(A[i,j]*B[i,j], C[i,j])

class TestChi(unittest.TestCase):
    # A random 2x2 quaternion matrix
    Q2 = np.zeros([2, 2], dtype=np.quaternion)
    Q2[0, 0] = np.quaternion(1, 0, 2, 0)
    Q2[1, 0] = np.quaternion(0, 1, 0, 1)
    Q2[0, 1] = np.quaternion(1, 2, 3, 4)
    Q2[1, 1] = np.quaternion(1, -2,0, 1)
    # A random 3x4 quaternion matrix
    Q34 = np.zeros([3, 4], dtype=np.quaternion)
    for i in range(3):
        for j in range(4):
            Q34[i, j] = np.quaternion(i, j, i+j, i-j)
    # A random 2x2 hermitian quaternion matrix
    QH = 0.5* (Q2 + Q2.conj().T)
    # A quaternionic identity matrix
    identity2 = np.zeros([2, 2], dtype=np.quaternion)
    identity2[0, 0] = np.quaternion(1, 0, 0, 0)
    identity2[1, 1] = np.quaternion(1, 0, 0, 0)

    def test_inverse(self):
        res = quaternion_matrix_inverse(self.Q2)
        mustbe_id = quatmatmul_matrix_by_matrix(res, self.Q2)
        mustbe_id_too = quatmatmul_matrix_by_matrix(self.Q2, res)
        np.testing.assert_array_almost_equal(
            quaternion.as_float_array(mustbe_id),
            quaternion.as_float_array(self.identity2)
            )
        np.testing.assert_array_almost_equal(
            quaternion.as_float_array(mustbe_id_too),
            quaternion.as_float_array(self.identity2)
            )

    def test_ifitruns(self):
        res = chi(self.Q2)

    def test_ifitruns_nonrectangular(self):
        res = chi(self.Q34)
        np.testing.assert_equal( res.shape, [6, 8])

    def test_zeros(self):
        Q = np.zeros([2, 2], dtype=np.quaternion)
        np.testing.assert_array_almost_equal( chi(Q), np.zeros([4, 4], dtype=complex))

    def test_identity(self):
        Q = np.zeros([2, 2], dtype=np.quaternion)
        Q[0, 0] = quaternion.one
        Q[1, 1] = quaternion.one
        np.testing.assert_array_almost_equal( chi(Q), np.eye(4, dtype=complex) )

    def test_chi_and_invchi_is_identity(self):
        Qc = chi(self.Q2)
        res = chi_inverse(Qc)
        self.assertTrue((self.Q2 == chi_inverse(Qc)).all())
        self.assertTrue((Qc == chi(chi_inverse(Qc))).all())

    def test_eigenvalues(self):
        with self.assertRaises(TypeError):
            np.linalg.eigvals(self.Q2)
        lam = np.linalg.eigvals(chi(self.QH))
        # Two things must happen:
        # a) Eigenvalues of chi should come in tuples of equal numbers
        # b) If input is hermitian, eigenvalues must be real
        np.testing.assert_array_almost_equal(np.imag(lam), np.zeros(4,))
        lam = np.sort(np.real(lam))
        self.assertAlmostEqual(lam[0], lam[1])
        self.assertAlmostEqual(lam[2], lam[3])

    def test_eigenvalues_mass(self):
        n = 3
        for i in range(10):
            np.random.seed(seed=i)
            A = np.random.rand(n, n, 4)
            Q = quaternion.as_quat_array(A)
            QH = Q + Q.conj().T
            lam = np.linalg.eigvals(chi(QH))
            np.testing.assert_array_almost_equal(np.imag(lam), np.zeros(2*n,))
            lam = np.sort(np.real(lam))
            for j in range(n):
                self.assertAlmostEqual(lam[2*j], lam[2*j+1])

    def test_eigenvalues_mass_2(self):
        n = 2
        for i in range(100):
            np.random.seed(seed=i)
            A = np.random.rand(n, n, 4)
            Q = quaternion.as_quat_array(A)
            QH = Q + Q.conj().T
            lam = np.linalg.eigvals(chi(QH))
            np.testing.assert_array_almost_equal(np.imag(lam), np.zeros(2*n,))
            lam = np.sort(np.real(lam))
            idiotimes = []
            for j in range(n):
                idiotimes.append(lam[2*j])
            orizousa = np.prod(idiotimes)
            ixnos = np.sum(idiotimes)
            self.assertAlmostEqual(orizousa, QH[0, 0]*QH[1,1] - QH[1,0]*QH[0,1])
            self.assertAlmostEqual(ixnos, QH[0,0]+QH[1,1])

    def test_eigenvectors(self):
        QH = self.QH
        L, V = eigdecomposition_hermitian(QH)
        for i in range(2):
            # Check that indeed they are eigenvectors
            diff = quatmatmul_matrix_by_vector(QH, V[i]) - V[i]*L[i]
            self.assertLessEqual(quatlength(quatdotproduct(diff, diff)), 1e-6)

    def test_eigenvectors_mass(self):
        n = 7
        for i in range(10):
            np.random.seed(seed=i)
            A = np.random.rand(n, n, 4)
            Q = quaternion.as_quat_array(A)
            QH = Q + Q.conj().T
            L, V = eigdecomposition_hermitian(QH)
            for i in range(n):
                # Check that indeed they are eigenvectors
                diff = quatmatmul_matrix_by_vector(QH, V[i]) - V[i]*L[i]
                self.assertLessEqual(quatlength(quatdotproduct(diff, diff)), 1e-6)

    def test_eigenvectors_nonhermitian(self):
        Q2 = self.Q2
        L, V = eigdecomposition(Q2, is_hermitian=False)
        for i in range(2):
            # Check that indeed they are eigenvectors
            diff = quatmatmul_matrix_by_vector(Q2, V[i]) - V[i]*L[i]
            self.assertLessEqual(quatlength(quatdotproduct(diff, diff)), 1e-6)

    def test_eigenvectors_mass_nonhermitian(self):
        n = 7
        for i in range(10):
            np.random.seed(seed=i)
            A = np.random.rand(n, n, 4)
            Q = quaternion.as_quat_array(A)
            L, V = eigdecomposition(Q, is_hermitian=False)
            for i in range(n):
                # Check that indeed they are eigenvectors
                diff = quatmatmul_matrix_by_vector(Q, V[i]) - V[i]*L[i]
                self.assertLessEqual(quatlength(quatdotproduct(diff, diff)), 1e-6)

if __name__ == '__main__':
    unittest.main()
