import cython
import numpy as np
cimport numpy as np

cpdef anisodiff(np.ndarray image, iteration=1, K=50, gamma=0.1):

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

    cdef np.ndarray flux
    cdef np.ndarray image_tmp

    for _ in range(iteration):
        image_tmp = np.zeros_like(image)
# Loop over dimensions        
        for i in range(image.ndim):
            flux = np.diff(image, axis=i, prepend=0)
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = 0
            flux[index] = 0
    
#            flux = flux * np.exp(-1/K * np.abs(flux))
            flux = flux * np.exp(-(flux/K)**2)
            flux = np.diff(flux, axis=i, append=0) 
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = -1
            flux[index] = 0

            image_tmp = image_tmp + gamma*flux

        image = image + image_tmp 
    
    return image
