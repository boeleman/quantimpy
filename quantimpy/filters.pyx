import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, abs

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

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef _anisodiff2D(np.ndarray[np.double_t, ndim=2] image, int option, int niter, float K, float gamma):
#
#    cdef int i
#    cdef int j
#
#    cdef np.ndarray[np.double_t, ndim=2] flux
#    cdef np.ndarray[np.double_t, ndim=2] image_tmp
#
#    for j in range(niter):
#        image_tmp = np.zeros_like(image)
## Loop over dimensions        
#        for i in range(image.ndim):
#            flux = np.diff(image, axis=i, prepend=0)
## Adiabatic boundary condition
#            index = [slice(None)]*flux.ndim
#            index[i] = 0
#            flux[tuple(index)] = 0
#
##            flux = flux * np.exp(-1/K * np.abs(flux))
#            if (option == 1):
#                flux = flux * np.exp(-(flux/K)**2.)
#            elif (option == 2):
#                flux = flux / (1. + (flux/K)**2.)
#
#            flux = np.diff(flux, axis=i, append=0) 
## Adiabatic boundary condition
#            index = [slice(None)]*flux.ndim
#            index[i] = flux.shape[i]-1
#            flux[tuple(index)] = 0
#
#            image_tmp = image_tmp + gamma*flux
#
#        image = image + image_tmp 
#
#    return image

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef _anisodiff3D(np.ndarray[np.double_t, ndim=3] image, int option, int niter, float K, float gamma):
#
#    cdef int i
#    cdef int j
#
#    cdef np.ndarray[np.double_t, ndim=3] flux
#    cdef np.ndarray[np.double_t, ndim=3] image_tmp
#
#    for j in range(niter):
#        image_tmp = np.zeros_like(image)
## Loop over dimensions        
#        for i in range(image.ndim):
#            flux = np.diff(image, axis=i, prepend=0)
## Adiabatic boundary condition
#            index = [slice(None)]*flux.ndim
#            index[i] = 0
#            flux[tuple(index)] = 0
#
##            flux = flux * np.exp(-1/K * np.abs(flux))
#            if (option == 1):
#                flux = flux * np.exp(-(flux/K)**2.)
#            elif (option == 2):
#                flux = flux / (1. + (flux/K)**2.)
#
#            flux = np.diff(flux, axis=i, append=0) 
## Adiabatic boundary condition
#            index = [slice(None)]*flux.ndim
#            index[i] = flux.shape[i]-1
#            flux[tuple(index)] = 0
#
#            image_tmp = image_tmp + gamma*flux
#
#        image = image + image_tmp 
#
#    return image

def _anisodiff2D(image, int option, int niter, float K, float gamma):

    dtype = np.double

    cdef double[:,:] image_view = np.asarray(image, dtype=dtype)

    flux = np.zeros_like(image, dtype=dtype)
    cdef double[:,::1] flux_view = flux
    
    tmp = np.zeros_like(image, dtype=dtype)
    cdef double[:,::1] tmp_view = tmp
    
    result = np.zeros_like(image, dtype=dtype)
    cdef double[:,::1] result_view = result

    cdef size_t x, y, xp, yp, x_max, y_max
    cdef int i

    x_max = image.shape[0]
    y_max = image.shape[1]

    result_view[:,:] = image_view[:,:]

    for i in range(niter):
        tmp_view[:,:] = 0

        for y in range(y_max):
            for x in range(x_max-1):
                xp = x+1
                flux_view[xp,y] = result_view[xp,y] - result_view[x,y]

            flux_view[0,y] = 0
            
            for x in range(x_max):
                if (option == 0):
                    flux_view[x,y] = flux_view[x,y] * exp(-1.*abs(flux_view[x,y])/K)
                elif (option == 1):
                    flux_view[x,y] = flux_view[x,y] * exp(-1.*(flux_view[x,y]/K)**2)
                elif (option == 2):
                    flux_view[x,y] = flux_view[x,y] / (1. + (flux_view[x,y]/K)**2)

            for x in range(x_max-1):
                xp = x+1
                flux_view[x,y] = flux_view[xp,y] - flux_view[x,y]

            flux_view[x_max-1,y] = 0

            for x in range(x_max):
                tmp_view[x,y] = tmp_view[x,y] + flux_view[x,y]

        for x in range(x_max):
            for y in range(y_max-1):
                yp = y+1
                flux_view[x,yp] = result_view[x,yp] - result_view[x,y]

            flux_view[x,0] = 0
            
            for y in range(y_max):
                if (option == 0):
                    flux_view[x,y] = flux_view[x,y] * exp(-1.*abs(flux_view[x,y])/K)
                elif (option == 1):
                    flux_view[x,y] = flux_view[x,y] * exp(-1.*(flux_view[x,y]/K)**2)
                elif (option == 2):
                    flux_view[x,y] = flux_view[x,y] / (1. + (flux_view[x,y]/K)**2)

            for y in range(y_max-1):
                yp = y+1
                flux_view[x,y] = flux_view[x,yp] - flux_view[x,y]

            flux_view[x,y_max-1] = 0

            for y in range(y_max):
                tmp_view[x,y] = tmp_view[x,y] + flux_view[x,y]

        for x in range(x_max):
            for y in range(y_max):
                result_view[x,y] = result_view[x,y] + gamma*tmp_view[x,y]

    return result


def _anisodiff3D(image, int option, int niter, float K, float gamma):

    dtype = np.double

    cdef double[:,:,:] image_view = np.asarray(image, dtype=dtype)

    flux = np.zeros_like(image, dtype=dtype)
    cdef double[:,:,::1] flux_view = flux
    
    tmp = np.zeros_like(image, dtype=dtype)
    cdef double[:,:,::1] tmp_view = tmp
    
    result = np.zeros_like(image, dtype=dtype)
    cdef double[:,:,::1] result_view = result

    cdef size_t x, y, z, xp, yp, zp, x_max, y_max, z_max
    cdef int i

    x_max = image.shape[0]
    y_max = image.shape[1]
    z_max = image.shape[2]

    result_view[:,:,:] = image_view[:,:,:]

    for i in range(niter):
        tmp_view[:,:] = 0

        for y in range(y_max):
            for z in range(z_max):
                for x in range(x_max-1):
                    xp = x+1
                    flux_view[xp,y,z] = result_view[xp,y,z] - result_view[x,y,z]

                flux_view[0,y,z] = 0
                
                for x in range(x_max):
                    if (option == 0):
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*abs(flux_view[x,y,z])/K)
                    elif (option == 1):
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*(flux_view[x,y,z]/K)**2)
                    elif (option == 2):
                        flux_view[x,y,z] = flux_view[x,y,z] / (1. + (flux_view[x,y,z]/K)**2)

                for x in range(x_max-1):
                    xp = x+1
                    flux_view[x,y] = flux_view[xp,y,z] - flux_view[x,y,z]

                flux_view[x_max-1,y,z] = 0

                for x in range(x_max):
                    tmp_view[x,y,z] = tmp_view[x,y,z] + flux_view[x,y,z]

        for z in range(z_max):
            for x in range(x_max):
                for y in range(y_max-1):
                    yp = y+1
                    flux_view[x,yp,z] = result_view[x,yp,z] - result_view[x,y,z]

                flux_view[x,0,z] = 0
                
                for y in range(y_max):
                    if (option == 0):
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*abs(flux_view[x,y,z])/K)
                    elif (option == 1):
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*(flux_view[x,y,z]/K)**2)
                    elif (option == 2):
                        flux_view[x,y,z] = flux_view[x,y,z] / (1. + (flux_view[x,y,z]/K)**2)

                for y in range(y_max-1):
                    yp = y+1
                    flux_view[x,y,z] = flux_view[x,yp,z] - flux_view[x,y,z]

                flux_view[x,y_max-1,z] = 0

                for y in range(y_max):
                    tmp_view[x,y,z] = tmp_view[x,y,z] + flux_view[x,y,z]

        for x in range(x_max):
            for y in range(y_max):
                for z in range(z_max-1):
                    zp = z+1
                    flux_view[x,y,zp] = result_view[x,y,zp] - result_view[x,y,z]

                flux_view[x,y,0] = 0
                
                for z in range(z_max):
                    if (option == 0):
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*abs(flux_view[x,y,z])/K)
                    elif (option == 1):                 
                        flux_view[x,y,z] = flux_view[x,y,z] * exp(-1.*(flux_view[x,y,z]/K)**2)
                    elif (option == 2):                 
                        flux_view[x,y,z] = flux_view[x,y,z] / (1. + (flux_view[x,y,z]/K)**2)

                for z in range(z_max-1):
                    zp = z+1
                    flux_view[x,y,z] = flux_view[x,y,zp] - flux_view[x,y,z]

                flux_view[x,y,z_max-1] = 0

                for z in range(z_max):
                    tmp_view[x,y,z] = tmp_view[x,y,z] + flux_view[x,y,z]

        for x in range(x_max):
            for y in range(y_max):
                for z in range(z_max):
                    result_view[x,y,z] = result_view[x,y,z] + gamma*tmp_view[x,y,z]

    return result


cpdef anisodiff(np.ndarray image, option=1, niter=1, K=50, gamma=0.1):
    if (image.ndim == 2):
        return _anisodiff2D(image.astype(np.double), option, niter, K, gamma)
    elif (image.ndim == 3):
        return _anisodiff3D(image.astype(np.double), option, niter, K, gamma)
    else:
        raise ValueError('Cannot handle more than three dimensions')
