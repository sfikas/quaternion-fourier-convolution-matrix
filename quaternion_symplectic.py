'''
Symplectic decomposition

References:
Ell and Sangwine, "Decomposition of 2D Hypercomplex Fourier Transforms into Pairs of Complex Fourier Transforms"

By G.Sfikas, Mar 2020
'''

from re import T
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import polar
import numpy as np
import unittest
from pyparsing import quoted_string
# This is the numpy-quaternion package. Already supports basic quaternion matrix operations.
import quaternion
#import tfquaternion
import quaternion_matrix, quaternion_circulant_matrix

def is_unitpure(x, tol=1e-6):
    assert(type(x) == np.quaternion)
    return(np.abs(x.real) < tol and np.abs(quaternion_matrix.quatlength(x) - 1) < tol)

def basis_change(q, basis, inverse_transform=False):
    '''
    Changes the basis of a single quaternion or a quaternion matrix.
    inverse_transform:  Use the basis to perform the inverse transform (working only for q=matrix)
    '''
    def scalarpart(x):
        assert(type(x) == np.quaternion)
        return(x.w)
    def vectorpart(x):
        assert(type(x) == np.quaternion)
        res = np.quaternion(0, x.x, x.y, x.z)
        return(res)
    if(type(q) == np.quaternion):
        raise NotImplementedError('Test not implemented yet.')
        if(inverse_transform==True):
            raise NotImplementedError('Basis change with inverse_transform=True is implemented only for quaternion matrix inputs.')
        res = np.quaternion(
            scalarpart(q),
            -0.5*(vectorpart(q)*basis[1] + basis[1]*vectorpart(q)),
            -0.5*(vectorpart(q)*basis[2] + basis[2]*vectorpart(q)),
            -0.5*(vectorpart(q)*basis[3] + basis[3]*vectorpart(q)),
        )
        return(res)
    elif(type(q) == np.ndarray and type(q.flat[0]) == np.quaternion):
        columnvector = False
        if(len(q.shape) == 1):
            # Treat the 1D as a special case of 2D (I had implemented 2D first..)
            columnvector = True
            q = q[..., np.newaxis]
        qsh0 = q.shape[0]
        qsh1 = q.shape[1]
        if(inverse_transform is False):
            T = np.linalg.inv(basis_change_matrix(basis))
        else:
            T = basis_change_matrix(basis)
        tt = T @ np.reshape(np.moveaxis(quaternion.as_float_array(q), [0, 1, 2], [2, 1, 0]), [4, -1])
        #ttAA = np.reshape(np.moveaxis(quaternion.as_float_array(q), [0, 1, 2], [2, 1, 0]), [4, -1])
        tt2 = np.moveaxis(np.reshape(tt, [4, qsh1, qsh0]), [0, 1, 2], [2, 1, 0])
        res = quaternion.as_quat_array(tt2)
        if columnvector:
            res = np.squeeze(res)
        return(res)
    else:
        try:
            print('Type of q is {}'.format(type(q.flat[0])))
        except:
            pass
        raise ValueError(
            'Input must be either a quaternion or a quaternion vector/matrix (was {}).'.format(
                type(q)
            ))
        

def basis_change_matrix(basis):
    '''
    b1-b4 are to be a set of quaternions that are going to play the role a new basis in R^4.
    Their components will be set to columns of a 4x4 matrix.

    The new coordinates are linked to standard basis coordinates by the following manner:
    standard_basis_coordinates = basis_change_matrix @ new_basis_coordinates
    '''
    assert(type(basis[0]) == np.quaternion)
    assert(type(basis[1]) == np.quaternion)
    assert(type(basis[2]) == np.quaternion)
    assert(type(basis[3]) == np.quaternion)
    res = np.zeros([4, 4], dtype=float)
    for i in range(4):
        res[0, i] = basis[i].w
        res[1, i] = basis[i].x
        res[2, i] = basis[i].y
        res[3, i] = basis[i].z
    return(res)          

def crossproduct(u, v):
    '''
    Cross product ("Εξωτερικό γινόμενο") for pure quaternions u,v.
    '''
    assert(type(u) == np.quaternion)
    assert(type(v) == np.quaternion)
    assert(u.w == 0)
    assert(v.w == 0)
    #this stems from the definition of Hamilton product in terms of S(u),V(u),S(v),V(v).        
    res = .5*(u*v - v*u)
    return(res)

def produce_valid_quaternionic_basis(mu1, mu2):
    '''
    From Ell and Sangwine, TIP 2007:
    Formula for Perpendicular Quaternion: The transform defines a plane in 3-space normal to the transform
    axis. *Any unit pure quaternion in this plane* is a valid choice for mu2. 
    We can obtain a choice for mu2 by making an arbitrary choice of of a third pure quaternion p, 
    which is not parallel to mu1 (p need not have unit modulus). 
    The cross product between the arbitrary quaternion p and mu1 must be perpendicular to mu1 (and p). 
    This satisfies one of the constraints on mu2 (its direction). 
    The remaining constraint (unit modulus) is easily satisfied by
    normalizing the cross product. [..] we select one of the three unit vectors i,j,k for p.
    '''
    if(mu1 is None):
        mu1 = np.quaternion(0, 1, 0, 0)
    assert(type(mu1) == np.quaternion)
    mu1 = mu1.normalized()
    if(mu2 is None):
        #If mu1 is parallel to i, pick k, so mu2=+-j
        if(mu1.normalized() == np.quaternion(0, 1, 0, 0) or mu1.normalized() == np.quaternion(0, -1, 0, 0)):
            mu2 = crossproduct(mu1, np.quaternion(0, 0, 0, 1))
        #If mu1 is not parallel to i, pick i
        else:
            mu2 = crossproduct(mu1, np.quaternion(0, 1, 0, 0))
    mu2 = mu2.normalized()
    assert(type(mu2) == np.quaternion)
    mu3 = (mu1 * mu2).normalized()
    return(mu1, mu2, mu3)
    

def symplectic_decomposition(A, mu1 = None, mu2 = None, save_true_values=False):
    '''
    Decomposes quaternionic matrix A into 
    
    simplex_part + perplex_part*mu2
    
    Both "simplex_part" and "perplex_part" are of the form: X+Y*mu1, where X,Y, are real matrices.

    A:      Input quaternionic array.
    mu1:    The symplectic decomposition axis.
    save_true_values:
            If false, save simplex part A1=X+Y*mu1 as the complex matrix A1_C=X+Y*i. Do likewise for the perplex part A2.
            If true, save simplex part as a quaternion matrix using its actual values (ie, dont map mu1->i). Do likewise for the perplex part A2.
            TODO: the name "save_true_values" is misleading, as the "perplex part" needs to be multiplied by mu2 before added to simplex to give the original value/matrix.

    The symplectic decomposition generalizes the Cayley-Dickson decomposition.
    The latter is equivalent to a symplectic decomposition with mu1 = i, mu2 = j.
    '''
    mu1, mu2, mu3 = produce_valid_quaternionic_basis(mu1, mu2)
    # Now convert inputs to the new basis: (1, mu1, mu2, mu3). This should be orthonormal (add check?)
    assert(type(A) == np.ndarray)
    assert(type(A.flat[0]) == np.quaternion)
    argand_plane_secondary_axis = mu1 if save_true_values else 1j
    basis = [
            np.quaternion(1, 0, 0, 0),
            mu1,
            mu2,
            mu3,
            ]
    Anewbasis = basis_change(A, basis=basis)
    B = quaternion.as_float_array(Anewbasis)
    simplex_part = B[..., 0] + B[..., 1]*argand_plane_secondary_axis
    perplex_part = B[..., 2] + B[..., 3]*argand_plane_secondary_axis
    '''
    if(len(A.shape) == 2):
            simplex_part = B[:, :, 0] + B[:, :, 1]*argand_plane_secondary_axis
            perplex_part = B[:, :, 2] + B[:, :, 3]*argand_plane_secondary_axis
    elif(len(A.shape) == 1):
            simplex_part = B[:, 0] + B[:, 1]*argand_plane_secondary_axis
            perplex_part = B[:, 2] + B[:, 3]*argand_plane_secondary_axis
    else:
        raise ValueError
    '''
    return(
        (simplex_part, perplex_part),
        (mu1, mu2, mu3)
        )


def symplectic_synthesis(simplex_part, perplex_part, mu1, mu2, mu3):
    assert(type(simplex_part) == np.ndarray)
    assert(type(perplex_part) == np.ndarray)
    assert(perplex_part.shape == simplex_part.shape)
    if(type(simplex_part.flat[0]) == np.quaternion or type(perplex_part.flat[0]) == np.quaternion):
        raise ValueError('Expecting simplex and perplex parts that are saved as real or complex matrices. If they are saved as their true values --ie form A+B*mu1-- simply perform simplex+perplex*mu2 to obtain them.')
    if(len(simplex_part.shape) == 2):
        sh0 = simplex_part.shape[0]
        sh1 = simplex_part.shape[1]
        tt = np.zeros([sh0, sh1, 4])
        tt[:, :, 0] = np.real(simplex_part)
        tt[:, :, 1] = np.imag(simplex_part)
        tt[:, :, 2] = np.real(perplex_part)
        tt[:, :, 3] = np.imag(perplex_part)
    elif(len(simplex_part.shape) == 1):
        sh0 = simplex_part.shape[0]
        tt = np.zeros([sh0, 4])
        tt[:, 0] = np.real(simplex_part)
        tt[:, 1] = np.imag(simplex_part)
        tt[:, 2] = np.real(perplex_part)
        tt[:, 3] = np.imag(perplex_part)        
    else:
        raise ValueError('Input simplex and perplex part should be 1D or 2D.')
    reconstructed = quaternion.as_quat_array(tt)
    res = basis_change(reconstructed, [
            np.quaternion(1, 0, 0, 0),
            mu1,
            mu2,
            mu3,
            ],
            inverse_transform=True
        )
    return(res)

def qfft(f, qft_axis, apply_shift):
    '''
    Computes a fast left-side quaternion fourier transfrom on signal f, with axis equal to qft_axis
    f:              Input quaternionic matrix.
    apply_shift:    If True, moves the lowest frequency to the center.
    '''
    if qft_axis is None:
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
    #if(qft_axis.real != 0 or qft_axis != qft_axis.normalized()):
    if(not is_unitpure(qft_axis)):
        raise ValueError('QFT transform axis should be a pure unit quaternion.')
    if len(f.shape) == 1:
        complex_transform = np.fft.fft
    elif len(f.shape) == 2:
        complex_transform = np.fft.fft2
    else:
        raise NotImplementedError('Input has more than two dimensions.')
    (f_simplex, f_perplex), (mu1, mu2, mu3) = symplectic_decomposition(f, mu1=qft_axis)
    F_simplex = complex_transform(f_simplex, norm='ortho')
    F_perplex = complex_transform(f_perplex, norm='ortho')
    if(apply_shift):
        F_simplex = np.fft.fftshift(F_simplex)
        F_perplex = np.fft.fftshift(F_perplex)
    F = symplectic_synthesis(F_simplex, F_perplex, mu1, mu2, mu3)
    return(F)

def qfft_right(f, qft_axis, apply_shift):
    '''
    Computes a fast *right-side* quaternion Fourier Transform.
    Uses the result from Moxey et al. 2003 / Ell and Sangwine et al. 2007.
    '''
    # This implements f'(x,y) = f(-x,-y)
    # Note that we need the 'roll', because e.g. f'(0,0)=f(0,0)
    if(len(f.shape) == 1):
        f_flipped = np.roll(np.flip(f), shift=1)
    elif(len(f.shape) == 2):
        f_flipped = np.roll(np.flip(f), axis=(0,1), shift=(1,1))
    else:
        raise NotImplementedError
    f_flipped_conjed = f_flipped.conj()
    F_tt = qfft(f_flipped_conjed, qft_axis=qft_axis, apply_shift=apply_shift)
    res = F_tt.conj()
    return(res)

def qfft_eigenvalues(f, qft_axis):
    '''
    Computes the "inbalanced" version of the QFT, where the forward transform bears
    no normalization constant.
    The result coincides to (part of) the left eigenvalues of the input.
    '''
    return(np.sqrt(np.prod(f.shape)) * qfft_right(f, qft_axis, apply_shift=False))

def iqfft_eigenvalues(f, qft_axis):
    '''
    The inverse qfft. Note that using qft_axis is inadequate, because we need to cancel out
    the normalization constant. See qfft_eigenvalues.
    '''
    return(qfft_right(f, -qft_axis, apply_shift=False) / np.sqrt(np.prod(f.shape)))

def iqfft(F, qft_axis, apply_shift):
    '''
    Computes a fast *left-side* quaternion inverse fourier transfrom on frequencies F, with axis equal to qft_axis
    F:              Input quaternionic matrix of signal frequencies.
    apply_shift:    If True, moves the lowest frequency *back from* the center to (0,0).
    '''
    if qft_axis is None:
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
    #if(qft_axis.real != 0 or qft_axis != qft_axis.normalized()):
    if(not is_unitpure(qft_axis)):
        raise ValueError('QFT transform axis should be a pure unit quaternion.')
    if len(F.shape) == 1:
        complex_transform = np.fft.ifft
    elif len(F.shape) == 2:
        complex_transform = np.fft.ifft2
    else:
        raise NotImplementedError('Input has more than two dimensions.')
    (F_simplex, F_perplex), (mu1, mu2, mu3) = symplectic_decomposition(F, mu1=qft_axis)
    if(apply_shift):
        F_simplex = np.fft.ifftshift(F_simplex)
        F_perplex = np.fft.ifftshift(F_perplex)
    f_simplex = complex_transform(F_simplex, norm='ortho')
    f_perplex = complex_transform(F_perplex, norm='ortho')
    f = symplectic_synthesis(f_simplex, f_perplex, mu1, mu2, mu3)
    return(f)

def load_image(X, fourth_channel=None):
    '''
    Loads image as a quaternionic matrix.
    Can select an additional greyscale input that will act as the fourth quaternion channel.
    '''
    if(type(X) == str):
        from PIL import Image
        X = np.array(Image.open(X))
    if(type(fourth_channel) == str):
        from PIL import Image
        fourth_channel = np.array(Image.open(fourth_channel))
    tt = np.zeros([X.shape[0], X.shape[1], 4])
    if(len(X.shape) == 2):
        #If grayscale, load input to real part
        tt[:, :, 0] = X
    elif(len(X.shape) == 3):
        #If multispectral, keep the first 3 channels as input
        tt[:, :, 1:] = X[:, :, :3]
    else:
        raise IOError('Invalid input size ({})'.format(X.shape))
    res = quaternion.as_quat_array(tt)
    if(fourth_channel is not None):
        raise NotImplementedError('Inputting a fourth channel is not implemented.')
    return(res)

def get_modulus_phase_axis(X, axis_convention=True, return_standard_complex_number=False):
    '''
    Principally used to visualize a quaternionic fourier transform.
    Uses the proposed visualization in Ell and Sangwine, IEEE TIP 2007, sec.V.,
    and the formulae for eigenaxis and eigenphase in Alexiadis and Darras, IEEE TMultimedia 2014.

    X:                  Quaternionic matrix input
    modulus:            A real matrix of quaternionic magnitudes.
    eigenphase:         theta values
    eigenaxis:          mu values
    axis_convention:    If True, this will make sure that the eigenphase covers [-pi,pi] (instead of being in [-pi/2,pi/2]).
                        By convention, the i-part of the axis will be chosen to be greater than or equal to 0.
                        This works as long as a xi function is picked so that xi^2[q] = 1.
    return_standard_complex_number:
                        This will return instead modulus and phase over the complex number (matrix) which
                        has a positive imaginary part and is similar to input number (matrix).
                        Instead of the axis argument, "None" will be returned.
    '''
    eps = 1e-8
    assert(type(X) == np.ndarray)
    if(type(X.flat[0]) == np.quaternion):
        X = quaternion.as_float_array(X)
    assert(X.shape[-1] == 4)
    # Compute axis_convention xi function (useful for better visualization practically)
    Xsign1 = np.sign(X[..., 1])
    Xsign2 = np.sign(X[..., 2])
    Xsign3 = np.sign(X[..., 3])
    xi = Xsign1
    xi[xi == 0] = Xsign2[xi == 0]
    xi[xi == 0] = Xsign3[xi == 0]
    xi[xi == 0] = 1 #No vector part if this applies
    # Compute magnitude of vector part
    absV = np.sqrt(X[..., 1]**2 + X[..., 2]**2 + X[..., 3]**2)
    if(return_standard_complex_number):
        return(X[..., 0], absV, None)
    # Compute modulus
    modulus = np.sqrt(X[..., 0]**2 + X[..., 1]**2 + X[..., 2]**2 + X[..., 3]**2)
    # Compute axis (mu)
    xshape_axis = list(X.shape)
    xshape_axis[-1] = 3
    axis = np.zeros(xshape_axis)
    for i in range(3):
        axis[..., i] = X[..., i+1] / (absV + eps)
        if axis_convention:
            axis[..., i] = xi * axis[..., i]
    # Compute phase (theta)
    #phase = np.arctan(absV / (X[..., 0] + eps))
    phase = np.arctan2(absV, X[..., 0] + eps)
    if axis_convention:
        phase = phase * xi
    return(modulus, phase, axis)

def vis(Q, visualization_style='polar', polar_style='standard', title=None):
    '''
    visualization_style:   polar, standardcomplex, asimage, channelwise
    '''
    from matplotlib import pyplot as plt
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, Normalize
    from matplotlib.cm import ScalarMappable
    # Visualize a quaternionic matrix
    if visualization_style == 'polar':
        fig = plt.figure(figsize=(20,20))
        modulus, phase, axis = get_modulus_phase_axis(Q)
        ax = plt.subplot(131)
        fig.suptitle(title, fontsize=15)        
        ax.set_title('log(1+Modulus)')
        mod = ax.imshow(np.log(1+modulus), cmap='gray')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        ax = plt.subplot(132)
        ax.set_title('Phase')
        mod = ax.imshow(phase, cmap='hsv') #Note, 'hsv' is a cyclic colormap used for *scalar* values
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        ax = plt.subplot(133)
        ax.set_title('Axis')
        xx = Normalize(vmin=-1, vmax=+1, clip=False)
        if polar_style == 'direction_cosines':
            # Use direction cosine 1 as Hue, direction cosine 2 as Saturation, and ignore dir.cosine 3
            # Cyclical for direction 1, but not cyclical for direction 2
            axis[:, :, -1] = 0
            axis_hsv = hsv_to_rgb(axis*.5+.5)
            mod = ax.imshow(axis_hsv)
            fig.colorbar(ScalarMappable(norm=xx, cmap='hsv'), ax=ax, fraction=0.046, pad=0.04)
        elif polar_style == 'standard':
            mod = ax.imshow(axis*.5+.5)
            #fig.colorbar(ScalarMappable(norm=xx, cmap='brg'), ax=ax, fraction=0.046, pad=0.04)
        elif polar_style == 'modulo':
            axis_mod = np.abs(np.mean(axis, -1))
            mod = ax.imshow(axis_mod, cmap='Set1')
            fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        else:
            raise ValueError('Undefined value for polar_style argument.')
        plt.show()
        return None
    elif visualization_style == 'standardcomplex':
        fig = plt.figure(figsize=(20,20))
        realpart, imaginarypart, _ = get_modulus_phase_axis(Q, return_standard_complex_number=True)
        ax = plt.subplot(121)
        fig.suptitle(title, fontsize=15)        
        ax.set_title('Real part')
        mod = ax.imshow(realpart, cmap='gray')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        ax = plt.subplot(122)
        fig.suptitle(title, fontsize=15)        
        ax.set_title('Imaginary part')
        mod = ax.imshow(imaginarypart, cmap='gray')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        plt.show()
    elif visualization_style == 'channelwise':
        fig = plt.figure(figsize=(20,20))
        fig.suptitle(title, fontsize=15)
        AA = quaternion.as_float_array(Q)
        ax = plt.subplot(141)
        ax.set_title('Real axis')
        mod = ax.imshow(AA[:, :, 0], cmap='gray')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        ax = plt.subplot(142)
        ax.set_title('Axis i')
        mod = ax.imshow(AA[:, :, 1], cmap='Reds')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)        
        ax = plt.subplot(143)
        ax.set_title('Axis j')
        mod = ax.imshow(AA[:, :, 2], cmap='Greens')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)        
        ax = plt.subplot(144)
        ax.set_title('Axis k')
        mod = ax.imshow(AA[:, :, 3], cmap='Blues')
        fig.colorbar(mod, ax=ax, fraction=0.046, pad=0.04)
        plt.show()
        return None
    elif visualization_style == 'asimage':
        #fig = plt.figure(figsize=(20,20))
        #fig.suptitle(title, fontsize=15)
        fig = None
        AA = quaternion.as_float_array(Q)
        B = np.zeros([AA.shape[0], AA.shape[1], 3], dtype=np.uint8)
        B[:, :, 0] = np.uint8(AA[:, :, 1])
        B[:, :, 1] = np.uint8(AA[:, :, 2])
        B[:, :, 2] = np.uint8(AA[:, :, 3])
        plt.imshow(B)
        plt.show()
        return None
    else:
        raise NotImplementedError
    return fig

class TestBasic(unittest.TestCase):
    def check_unitpure(self):
        self.assertTrue(is_unitpure(np.quaternion(0, 1, 0, 0)))
        self.assertTrue(is_unitpure(np.quaternion(0, 0, 1, 0)))
        self.assertFalse(is_unitpure(np.quaternion(1, 0, 0, 0)))

    def test_selftest(self):
        # A 3x4 quaternion matrix
        Q34 = np.zeros([7, 10], dtype=np.quaternion)
        for i in range(Q34.shape[0]):
            for j in range(Q34.shape[1]):
                Q34[i, j] = np.quaternion(i, j, i+j, i-j)
        (simplex, perplex), (mu1, mu2, _) = symplectic_decomposition(Q34, mu1=np.quaternion(0, 1, 0, 0), mu2=np.quaternion(0, 0, 1, 0))
        ## Now reconstruct the original image
        tt = symplectic_synthesis(simplex, perplex, mu1=np.quaternion(0, 1, 0, 0), mu2=np.quaternion(0, 0, 1, 0), mu3=mu1*mu2)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(tt), quaternion.as_float_array(Q34))

    def test_selftest_2(self):
        # A 3x4 quaternion matrix
        Q34 = np.zeros([7, 10], dtype=np.quaternion)
        for i in range(Q34.shape[0]):
            for j in range(Q34.shape[1]):
                Q34[i, j] = np.quaternion(i, j, i+j, i-j)
        (simplex, perplex), (mu1, mu2, mu3) = symplectic_decomposition(Q34, mu1=np.quaternion(0, 2, 3, 4))
        (simplex_t, perplex_t), (mu1, mu2, mu3) = symplectic_decomposition(Q34, mu1=np.quaternion(0, 2, 3, 4), save_true_values=True)
        tt = symplectic_synthesis(simplex, perplex, mu1=mu1, mu2=mu2, mu3=mu3)
        tt_2 = simplex_t + perplex_t*mu2
        #np.testing.assert_array_almost_equal(quaternion.as_float_array(tt), quaternion.as_float_array(Q34))
        np.testing.assert_array_almost_equal(quaternion.as_float_array(tt), quaternion.as_float_array(tt_2))

    def test_check_change_basis(self):
        T = basis_change_matrix([
            np.quaternion(1, 0, 0, 0),
            np.quaternion(0, 1, 0, 0),
            np.quaternion(0, 0, 1, 0),
            np.quaternion(0, 0, 0, 1),
        ])
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_check_change_basis2(self):
        Q34 = np.zeros([2, 3], dtype=np.quaternion)
        for i in range(Q34.shape[0]):
            for j in range(Q34.shape[1]):
                Q34[i, j] = np.quaternion(i, j, i+j, i-j)
        mu1 = np.quaternion(0, 1, 1, 1).normalized()
        mu2 = (np.quaternion(0, 0, 1, 0) - np.quaternion(0, 0, 0, 1)).normalized()
        newbasis = [
            np.quaternion(1, 0, 0, 0),
            mu1,
            mu2,
            mu1*mu2,
        ]
        newQ = basis_change(Q34, newbasis)
        np.testing.assert_array_equal(newQ.shape, Q34.shape)
        newQ2 = basis_change(newQ, newbasis, inverse_transform=True)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(newQ2), quaternion.as_float_array(Q34))

    def test_get_modulus_phase_axis(self):
        Q = np.zeros([100, 150], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(i, j, i+j, i-j)
        Q_qftd = qfft(Q, np.quaternion(0, 1, 0, 0), apply_shift=True)
        m, p, a = get_modulus_phase_axis(Q_qftd)
        self.assertEqual(m.shape, (100, 150))
        self.assertEqual(p.shape, (100, 150))
        self.assertEqual(a.shape, (100, 150, 3))                

    def test_get_modulus_phase_axis_2(self):
        sh0 = 30
        sh1 = 40
        Q = np.zeros([sh0, sh1], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(i, j, i+j, i-j)
        m, p, a = get_modulus_phase_axis(Q, axis_convention=True)
        qa = np.zeros([sh0, sh1, 4])
        qa[:, :, 1:] = a
        qa2 = quaternion.as_quat_array(qa)
        xx = m * (np.cos(p) + qa2*np.sin(p))
        ###
        # Back to float arrays
        f1 = quaternion.as_float_array(Q)
        f2 = quaternion.as_float_array(xx)
        np.testing.assert_array_almost_equal(f1, f2)

    def test_get_modulus_phase_axis_3(self):
        sh0 = 3
        sh1 = 4
        Q = np.zeros([sh0, sh1], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(1, 0, -i*2, 5*i-j)
        m, p, a = get_modulus_phase_axis(Q, axis_convention=False)
        qa = np.zeros([sh0, sh1, 4])
        qa[:, :, 1:] = a
        qa2 = quaternion.as_quat_array(qa)
        xx = m * (np.cos(p) + qa2*np.sin(p))
        ###
        # Back to float arrays
        f1 = quaternion.as_float_array(Q)
        f2 = quaternion.as_float_array(xx)
        np.testing.assert_array_almost_equal(f1, f2)



class TestQFT(unittest.TestCase):
    def test_qft(self):
        Q = np.zeros([100, 150], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(i, j, i+j, i-j)
        Q_qftd = qfft(Q, np.quaternion(0, 1, 0, 0), apply_shift=True)
        Qrecon = iqfft(Q_qftd, np.quaternion(0, 1, 0, 0), apply_shift=True)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(Q), quaternion.as_float_array(Qrecon))

    def test_qft_noshift(self):
        Q = np.zeros([100, 150], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(i, j, i+j, i-j)
        Q_qftd = qfft(Q, np.quaternion(0, 1, 0, 0), apply_shift=False)
        Qrecon = iqfft(Q_qftd, np.quaternion(0, 1, 0, 0), apply_shift=False)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(Q), quaternion.as_float_array(Qrecon))

    def test_qft_2(self):
        # IQFT with axis mu must be equal to QFT with axis -mu.
        Q = np.zeros([100, 150], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = np.quaternion(i, j, i+j, i-j)
        #
        mu = np.quaternion(0, 1, 0, 0)
        Q1 = qfft(Q, qft_axis=+mu, apply_shift=False)
        Q2 = iqfft(Q, qft_axis=-mu, apply_shift=False)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(Q1), quaternion.as_float_array(Q2))

    def test_qft1D_2(self):
        # IQFT with axis mu must be equal to QFT with axis -mu.
        Q = np.zeros([5, ], dtype=np.quaternion)
        for i in range(Q.shape[0]):
            Q[i] = np.quaternion(i, 2*i, 0, -i)
        mu = np.quaternion(0, 1, 2, -1).normalized()
        Q1 = qfft(Q, qft_axis=+mu, apply_shift=False)
        Q2 = iqfft(Q, qft_axis=-mu, apply_shift=False)
        np.testing.assert_array_almost_equal(quaternion.as_float_array(Q1), quaternion.as_float_array(Q2))

    def test_quaternion_fourier_matrix_performs_qft(self):
        N = 200
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
        Q = quaternion_circulant_matrix.create_qft_matrix(N, axis=qft_axis)
        x = np.zeros([N,], dtype=np.quaternion)
        for i in range(N):
            x[i] = np.quaternion(i, -i, 2, 5*i+3)
        res1 = quaternion_matrix.quatmatmul_matrix_by_vector(Q, x)
        res2 = qfft(x, qft_axis=qft_axis, apply_shift=False)
        self.assertAlmostEqual(np.sum(np.abs(res1-res2)), 0)

    def test_quaternion_fourier_matrix_performs_qft_2(self):
        #Non-equality because the two transforms dont use the same axis.
        N = 200
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
        qft_axis2 = np.quaternion(0, 1, 0, 0).normalized()
        Q = quaternion_circulant_matrix.create_qft_matrix(N, axis=qft_axis)
        x = np.zeros([N,], dtype=np.quaternion)
        for i in range(N):
            x[i] = np.quaternion(i, -i, 2, 5*i+3)
        res1 = quaternion_matrix.quatmatmul_matrix_by_vector(Q, x)
        res2 = qfft(x, qft_axis=qft_axis2, apply_shift=False)
        self.assertNotAlmostEqual(np.sum(np.abs(res1-res2)), 0)

    def test_quaternion_fourier_matrix_performs_qft_2(self):
        #Non-equality because the two transforms use axes that one is the opposite of the other.
        N = 200
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
        Q = quaternion_circulant_matrix.create_qft_matrix(N, axis=qft_axis)
        x = np.zeros([N,], dtype=np.quaternion)
        for i in range(N):
            x[i] = np.quaternion(i, -i, 2, 5*i+3)
        res1 = quaternion_matrix.quatmatmul_matrix_by_vector(Q, x)
        res2 = qfft(x, qft_axis=-qft_axis, apply_shift=False)
        self.assertNotAlmostEqual(np.sum(np.abs(res1-res2)), 0)

    def test_rightqft(self):
        '''
        The implementation of qfft_right uses a result from Moxey et al. 2003/Ell and Sangwine 2007.
        '''
        N = 50
        qft_axis = np.quaternion(0, 1, 1, 1).normalized()
        Q = quaternion_circulant_matrix.create_qft_matrix(N, axis=qft_axis)
        x = np.zeros([N,], dtype=np.quaternion)
        for i in range(N):
            x[i] = np.quaternion(i, -i, 2, 5*i+3)
        #FQ_left = quatmatmul_matrix_by_vector(Q, x, multiplyfromright=True)
        FQ_right = quaternion_matrix.quatmatmul_matrix_by_vector(Q, x, multiplyfromright=False)
        A = qfft_right(x, qft_axis=qft_axis, apply_shift=False)
        self.assertAlmostEqual(np.sum(np.abs(FQ_right - A)), 0)

    def test_propo35_circulant_matrix_eigenvectors(self):
        '''
        This tests the proposition 3.5, first part.
        '''
        N = 100 
        mykernel = np.zeros([N, ], dtype=np.quaternion)
        for i in range(N):
            mykernel[i] = np.quaternion(i, -i, 2*i+5, np.cos(i)) #random values as kernel
        C = quaternion_circulant_matrix.quaternionic_circulant_filter(mykernel)
        mu = np.quaternion(0, 1, 1, 1).normalized()
        Q = quaternion_circulant_matrix.create_qft_matrix(N, axis=-mu)
        lam = np.sqrt(N) * qfft_right(mykernel, qft_axis=mu, apply_shift=False)
        shouldbe_zerosum = 0
        for i in range(N):
            column_1 = Q[:, i]
            tt_1 = quaternion_matrix.quatmatmul_matrix_by_vector(C, column_1)
            tt_2 = lam[i] * column_1
            shouldbe_zerosum += np.sum(np.abs(tt_1-tt_2))
        self.assertAlmostEqual(shouldbe_zerosum, 0)

class TestConvolutionTheorem(unittest.TestCase):
    def test_1d_product(self):
        N = 5
        filter = np.zeros([N,], dtype=np.quaternion)
        v = np.zeros([N,], dtype=np.quaternion)
        for i in range(N):
            filter[i] = np.quaternion(i, i*2, 1, 5*i-3)
            #filter[i] = np.quaternion(1, 0, 0, 0)
        v[i] = np.quaternion(-i, i, i, 1)
        C = quaternion_circulant_matrix.quaternionic_circulant_filter(filter)
        mu = np.quaternion(0, 10, 0, -4).normalized()
        result_1 = quaternion_circulant_matrix.quaternion_circulant_by_vector_product(C=C, Ckernel=None, Ceigenvalues_forward=None, Ceigenvalues_backward=None, v=v)
        result_2 = quaternion_circulant_matrix.quaternion_circulant_by_vector_product(C=None, Ckernel=filter, Ceigenvalues_forward=None, Ceigenvalues_backward=None, v=v,
                mu=mu)
        result_3 = quaternion_circulant_matrix.quaternion_circulant_by_vector_product(C=None, Ckernel=None, 
                        Ceigenvalues_forward = np.sqrt(N) * qfft_right(filter, qft_axis=+mu, apply_shift=False),
                        Ceigenvalues_backward = np.sqrt(N) * qfft_right(filter, qft_axis=-mu, apply_shift=False),
                        v=v, mu=mu)
        for i in range(len(filter)):
            self.assertAlmostEqual(result_1[i], result_2[i])
            self.assertAlmostEqual(result_1[i], result_3[i])


    def test_2d_product(self):
        '''
        #TODO
        M = 5
        N = 6
        filter = np.zeros([M,N], dtype=np.quaternion)
        v = np.zeros([M,N], dtype=np.quaternion)
        for i in range(M):
            for j in range(N):
                filter[i,j] = np.quaternion(i*j, j+i*2, 1, 5*i-3)
        v[i] = np.quaternion(-i, i, i, 1)
        #C = quaternion_circulant_matrix.quaternionic_circulant_filter(filter)
        mu = np.quaternion(0, 10, 0, -4).normalized()
        #result_1 = quaternion_circulant_matrix.quaternion_circulant_by_vector_product(C=C, Ckernel=None, Ceigenvalues_forward=None, Ceigenvalues_backward=None, v=v)
        result_2 = quaternion_circulant_matrix.quaternion_doublyblockCirculant_by_vector_product(C=None, Ckernel=filter, Ceigenvalues_forward=None, Ceigenvalues_backward=None, v=v,
                mu=mu)
        result_3 = quaternion_circulant_matrix.quaternion_doublyblockCirculant_by_vector_product(C=None, Ckernel=None, 
                        Ceigenvalues_forward = np.sqrt(M*N) * qfft_right(filter, qft_axis=+mu, apply_shift=False),
                        Ceigenvalues_backward = np.sqrt(M*N) * qfft_right(filter, qft_axis=-mu, apply_shift=False),
                        v=v, mu=mu)
        print(result_2)
        print(result_3)
        for i in range(len(filter)):
            #self.assertAlmostEqual(result_1[i], result_2[i])
            self.assertAlmostEqual(result_2[i], result_3[i])
        '''
        pass

if __name__ == '__main__':
    unittest.main()
