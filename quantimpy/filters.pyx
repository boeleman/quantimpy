import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp

###############################################################################
#
#Original Matlab code:
#
#@incollection{perona1994,
#	author = {Perona, Pietro and Shiota, Takahiro and Malik, Jitendra},
#	booktitle = {{Geometry-driven diffusion in computer vision}},
#	editor = {{ter Haar Romeny}, Bart M.},
#	isbn = {9789401716994},
#	pages = {73--92},
#	publisher = {Springer},
#	title = {{Anisotropic diffusion}},
#	year = {1994}
#}
#
#function [outimage] = anisodiff(inimage,iteration,K)
#
#lambda = 0.25;
#outimage = inimage;     [m,n] = size(inimage);
#
#rowC = [1:m];           rowN = [1 1:m-1];           rowS = [2:m m];
#colC = [1:n];           colE = [1 1:n-1];           colW = [2:n n];
#
#for i=1:iterations,
#    deltaN = outimage(rowN,colC) - outimage(rowC,colC);
#    deltaE = outimage(rowC,colE) - outimage(rowC,colC);
#
#    fluxN = deltaN .* exp( - (1/K) * abs(deltaN) );
#    fluxE = deltaE .* exp( - (1/K) * abs(deltaE) );
#
#    outimage = outimage + lambda * 
#    (fluxN - fluxN(rowS,colC) + fluxE - fluxE(rowC,colW));
#end;
#
###############################################################################

ctypedef fused my_type:
    unsigned char
    unsigned short
    unsigned int
    double
    char
    short
    int

@cython.boundscheck(False)
@cython.wraparound(False)
def _anisodiff2D(my_type[:,::1] image, int option, int niter, double K, double gamma):

    cdef int i, j
    cdef int x_max, y_max

    cdef double Kinv = 1./K**2

    cdef np.ndarray[np.float64_t, ndim=2] flux
    cdef np.ndarray[np.float64_t, ndim=2] result
    cdef np.ndarray[np.float64_t, ndim=2] result_tmp
    
    result = np.asarray(image, dtype=np.float64) 
    result_tmp = np.zeros_like(result)

    for j in range(niter):
        result_tmp[:,:] = 0
# Loop over dimensions        
        for i in range(2):
            flux = np.diff(result, axis=i, prepend=0)
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = 0
            flux[tuple(index)] = 0

            if (option == 1):
                flux = flux * np.exp(-flux*flux*Kinv)
            elif (option == 2):
                flux = flux / (1. + (flux*flux*Kinv))

            flux = np.diff(flux, axis=i, append=0) 
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = flux.shape[i]-1
            flux[tuple(index)] = 0

            result_tmp = result_tmp + gamma*flux

        result = result + result_tmp

# Normalize between -1 and 1
    return result/np.amax(np.abs(result))

@cython.boundscheck(False)
@cython.wraparound(False)
def _anisodiff3D(my_type[:,:,::1] image, int option, int niter, double K, double gamma):

    cdef int i, j
    cdef int x_max, y_max

    cdef double Kinv = 1./K**2

    cdef np.ndarray[np.float64_t, ndim=3] flux
    cdef np.ndarray[np.float64_t, ndim=3] result
    cdef np.ndarray[np.float64_t, ndim=3] result_tmp
    
    result = np.asarray(image, dtype=np.float64) 
    result_tmp = np.zeros_like(result)

    for j in range(niter):
        result_tmp[:,:,:] = 0.0
# Loop over dimensions        
        for i in range(3):
            flux = np.diff(result, axis=i, prepend=0)
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = 0
            flux[tuple(index)] = 0

            if (option == 1):
                flux = flux * np.exp(-flux*flux*Kinv)
            elif (option == 2):
                flux = flux / (1. + (flux*flux*Kinv))

            flux = np.diff(flux, axis=i, append=0) 
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = flux.shape[i]-1
            flux[tuple(index)] = 0

            result_tmp = result_tmp + gamma*flux

        result = result + result_tmp

# Normalize between -1 and 1
    return result/np.amax(np.abs(result))

cpdef anisodiff(image, option=1, niter=1, K=50, gamma=0.1):
    image = np.ascontiguousarray(image)
    if (image.ndim == 2):
        return _anisodiff2D(image, option, niter, K, gamma)
    elif (image.ndim == 3):
        return _anisodiff3D(image, option, niter, K, gamma)
    else:
        raise ValueError('Cannot handle more than three dimensions')
