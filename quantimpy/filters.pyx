r"""Filters for image processing

This module contains filters and other functions for image processing and
thresholding on both 2D and 3D Numpy [1]_ arrays.
"""

import cython
import numpy as np
from numpy.linalg import norm
cimport numpy as np

# Scikit image data types
#uint8
#uint16
#uint32
#float
#int8
#int16
#int32
ctypedef fused my_type:
    unsigned char
    unsigned short
    unsigned int
    double
    signed char
    signed short
    signed int

@cython.boundscheck(False)
@cython.wraparound(False)
def _anisodiff2D(my_type[:,::1] image, int option, int niter, double K, double gamma):

    cdef int i, j
    cdef int x_max, y_max

    cdef double K_inv = 1./K
    cdef double K2_inv = 1./K**2

    cdef np.ndarray[np.float64_t, ndim=2] flux
    cdef np.ndarray[np.float64_t, ndim=2] result
    cdef np.ndarray[np.float64_t, ndim=2] result_tmp
    
    result = np.asarray(image, dtype=np.float64) 
    result_tmp = np.zeros_like(result)

    dtype = None
    cdef double dtype_inv = 1.0 
    if my_type == "unsigned char":
        dtype = np.uint8
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "unsigned short":
        dtype = np.uint16
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "unsigned int":
        dtype = np.uint32
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "double":
        dtype = np.float64
        dtype_inv = 1.0 
    elif my_type == "signed char":
        dtype = np.int8
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "signed short":
        dtype = np.int16
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "signed int":
        dtype = np.int32
        dtype_inv = 1.0/(np.iinfo(dtype).max)

    for j in range(niter):
        result_tmp[:,:] = 0
# Loop over dimensions        
        for i in range(2):
            flux = np.diff(result, axis=i, prepend=0)
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = 0
            flux[tuple(index)] = 0

            if (option == 0):
                flux = flux * np.exp(-np.abs(flux)*K_inv)
            elif (option == 1):
                flux = flux * np.exp(-flux**2*K2_inv)
            elif (option == 2):
                flux = flux / (1. + (flux**2*K2_inv))

            flux = np.diff(flux, axis=i, append=0) 
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = flux.shape[i]-1
            flux[tuple(index)] = 0

            result_tmp = result_tmp + gamma*flux

        result = result + result_tmp

# Normalize between -1 and 1 or 0 and 1
    return result*dtype_inv

@cython.boundscheck(False)
@cython.wraparound(False)
def _anisodiff3D(my_type[:,:,::1] image, int option, int niter, double K, double gamma):

    cdef int i, j
    cdef int x_max, y_max

    cdef double K_inv = 1./K
    cdef double K2_inv = 1./K**2

    cdef np.ndarray[np.float64_t, ndim=3] flux
    cdef np.ndarray[np.float64_t, ndim=3] result
    cdef np.ndarray[np.float64_t, ndim=3] result_tmp
    
    result = np.asarray(image, dtype=np.float64) 
    result_tmp = np.zeros_like(result)

    dtype = None
    cdef double dtype_inv = 1.0 
    if my_type == "unsigned char":
        dtype = np.uint8
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "unsigned short":
        dtype = np.uint16
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "unsigned int":
        dtype = np.uint32
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "double":
        dtype = np.float64
        dtype_inv = 1.0 
    elif my_type == "signed char":
        dtype = np.int8
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "signed short":
        dtype = np.int16
        dtype_inv = 1.0/(np.iinfo(dtype).max)
    elif my_type == "signed int":
        dtype = np.int32
        dtype_inv = 1.0/(np.iinfo(dtype).max)

    for j in range(niter):
        result_tmp[:,:,:] = 0.0
# Loop over dimensions        
        for i in range(3):
            flux = np.diff(result, axis=i, prepend=0)
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = 0
            flux[tuple(index)] = 0

            if (option == 0):
                flux = flux * np.exp(-np.abs(flux)*K_inv)
            elif (option == 1):
                flux = flux * np.exp(-flux**2*K2_inv)
            elif (option == 2):
                flux = flux / (1. + (flux**2*K2_inv))

            flux = np.diff(flux, axis=i, append=0) 
# Adiabatic boundary condition
            index = [slice(None)]*flux.ndim
            index[i] = flux.shape[i]-1
            flux[tuple(index)] = 0

            result_tmp = result_tmp + gamma*flux

        result = result + result_tmp

# Normalize between -1 and 1 or 0 and 1
    return result*dtype_inv

@cython.binding(True)
cpdef anisodiff(image, option=1, niter=1, K=50, gamma=0.1):
    r"""
    Anisotropic diffusion filter

    This function applies an anisotropic diffusion filter to the 2D and 3D Numpy
    array `image`. This is also known as Perona Malik diffusion [2]_. This is an
    edge preserving noise reduction method. The code is based on a Matlab code
    by Perona, Shiota, and Malik [3]_.

    Parameters
    ----------
    image : ndarray, {int, uint, float}
        Either 2D or 3D grayscale input image.
    option : int, defaults to 1
        The `option` parameter selects the conduction coefficient used by the
        filter. `option=0` selects the following conduction coefficient: 

        .. math:: g (\nabla I) = \exp{(-\frac{||\nabla I||}{K})},

        where :math:`\nabla I` is the image brightness gradient, and :math:`K`
        is a constant. The above equation is used in a Matlab code by Perona,
        Shiota, and Malik [3]_. `option=1` selects the conduction coefficient: 

        .. math:: g (\nabla I) = \exp{(-\left(\frac{||\nabla I||}{K}\right)^{2})},

        and `option=2` selects the coefficient: 

        .. math:: g (\nabla I) = \frac{1}{1 + (\frac{||\nabla I||}{K})^{2}}.

        Option one privileges high-contrast edges over low-contrast ones, while
        the second option privileges wide regions over smaller ones [2]_.
    niter : int, defaults to 1
        The number of iterations that the filter is applied.
    K : float, defaults to 50
        The value of constant :math:`K` in the above equations.
    gamma : float, defaults to 0.1
        Sets the diffusion "time" step size. When :math:`\gamma \leq 0.25`,
        stability is ensured. 
            
    Returns
    -------
    out : ndarray, float
        The noise reduced 2D or 3D output image. The return data type is float
        and the image is normalized betweeen 0 and 1 or -1 and 1.

    See Also
    --------

    Examples
    --------
    This example uses the scikit-image Python package [4]_, the Matplotlib Python
    package [5]_, and the SciPy Python package [6]_.

    .. code-block:: python

        import matplotlib.pyplot as plt
        from scipy import misc
        from skimage.util import random_noise
        from quantimpy import filters

        # Create image with noise
        image = misc.ascent()
        image = image.astype("uint8") # Fix data type
        image = random_noise(image, mode='speckle', mean=0.1)

        # Filter image
        result = filters.anisodiff(image, niter=5)

        # Show results
        fig = plt.figure()
        plt.gray()  # show the filtered result in grayscale
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        ax1.imshow(image)
        ax2.imshow(result)
        plt.show()
    """
    image = np.ascontiguousarray(image)
    if (image.ndim == 2):
        return _anisodiff2D(image, option, niter, K, gamma)
    elif (image.ndim == 3):
        return _anisodiff3D(image, option, niter, K, gamma)
    else:
        raise ValueError('Cannot handle more than three dimensions')

@cython.binding(True)
def histogram(image, int bits=8):
    r"""
    Create an image histogram

    This function creates an histogram for the 2D or 3D Numpy array `image`. The
    histogram is 8-bit (:math:`2^8` bins) by default. The function is coded
    around the `numpy.histogram` function. However, the functions returns the
    center locations of the bins instead of the edges. For `float` or 16-bit
    images the bin size is scaled accordingly.

    Parameters
    ----------
    image : ndarray, {int, uint, float}
        Either 2D or 3D grayscale input image.
    bits : int, defaults to 8
        :math:`2^{\text{bits}}` bins are used for the histogram. Defaults to 8
        bits or 256 bins.
    
    Returns
    -------
    out : tuple, float
        Returns two ndarrays: one with the histogram and one with the bin
        centers.    

    See Also
    --------
    ~quantimpy.filters.unimodal

    Examples
    --------
    This example uses the Matplotlib Python package [5]_, and the SciPy Python
    package [6]_.

    .. code-block:: python

        import matplotlib.pyplot as plt
        from scipy import misc
        from quantimpy import filters

        # Create image
        image = misc.ascent()
        image = image.astype("uint8") # Fix data type

        # Compute histpgram
        hist, bins = filters.histogram(image)
        
        # Plot histogram
        plt.bar(bins,hist)
        plt.show()

    """
    cdef double dtype_min
    cdef double dtype_max
    
    bits = 2**bits

    if (image.dtype == "float64"):
        if (np.amin(image) < 0.0):
            dtype_min = -1.0 - 1.0/float(bits-1)
            dtype_max =  1.0 + 1.0/float(bits-1)
        else:
            dtype_min = 0.0 - 0.5/float(bits-1)
            dtype_max = 1.0 + 0.5/float(bits-1)
    else:                    
        dtype_min = np.iinfo(image.dtype).min
        dtype_max = np.iinfo(image.dtype).max
        dtype_delta = dtype_max - dtype_min
        if (dtype_min < 0.0):
            dtype_max = dtype_min + (float(bits) - 0.5) * float(dtype_delta)/float(bits-1)
            dtype_min = dtype_min - 0.5 * float(dtype_delta)/float(bits-1)
        else:
            dtype_min = dtype_min - 0.5*dtype_max/float(bits-1)
            dtype_max = dtype_max + 0.5*dtype_max/float(bits-1)

# Compute 8 bit histogram
    hist, bins = np.histogram(image, range=(dtype_min, dtype_max), bins=bits)
# Find middle values
    bins = 0.5*(bins[1:] + bins[:-1])

    return hist, bins

@cython.binding(True)
def unimodal(np.ndarray[np.int64_t, ndim=1] hist):
    r"""
    Compute unimodal threshold

    Using image histogram `hist`, this function computes the unimodal threshold
    [7]_. This algorithms is slightly modified modified from the original
    method. Instead of defining the end of the distribution as the point where
    the histogram is zero, this algorithm takes the point that contains 99.7% of
    the observations. This is equivalent to three times the standard deviation
    for a Gaussian distribution.
    
    Parameters
    ----------
    hist : ndarray, int
        Histogram computed by the function :func:`~quantimpy.filters.histogram`.
    
    Returns
    -------
    out : int
        Index of the unimodal threshold value in input array `hist`.

    See Also
    --------
    ~quantimpy.filters.histogram

    Examples
    --------
    This example uses the Matplotlib Python package [5]_, and the SciPy Python
    package [6]_.

    .. code-block:: python

        import matplotlib.pyplot as plt
        from scipy import misc
        from scipy import ndimage
        from quantimpy import filters

        # Create 8uint image
        image = misc.ascent()
        image = image.astype("uint8") # Fix data type

        # Filter image
        result = filters.anisodiff(image)

        # Edge detection
        laplace = ndimage.laplace(result)

        # Compute histpgram
        hist, bins = filters.histogram(laplace)

        # Compute unimodal threshold
        thrshld = filters.unimodal(hist)

        # Plot histogram
        plt.bar(bins, hist, width=5e-3)
        plt.scatter(bins[thrshld], hist[thrshld])
        plt.show()

    References
    ----------
    .. [1] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt et al.,
        "Array programming with NumPy", Nature, vol. 585, pp 357-362, 2020,
        doi:`10.1038/s41586-020-2649-2`_

    .. _10.1038/s41586-020-2649-2: https://doi.org/10.1038/s41586-020-2649-2

    .. [2] Pietro Perona and Jitendra Malik, "Scale-space and edge detection
        using anisotropic diffusion", IEEE Transactions on pattern analysis and
        machine intelligence, vol. 12, no. 7, pp 629-639, 1990, doi:`10.1109/34.56205`_

    .. _10.1109/34.56205: https://doi.org/10.1109/34.56205

    .. [3] Pietro Perona, Takahiro Shiota, and Jitendra Malik, "Anisotropic
        diffusion", in "Geometry-driven diffusion in computer vision", ed. Bart
        M. ter Haar Romeny, pp 73-92, 1994, isbn: 9789401716994

    .. [4] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
        François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart,
        Tony Yu and the scikit-image contributors. "scikit-image: Image
        processing in Python." PeerJ 2:e453 (2014) doi: `10.7717/peerj.453`_

    .. _10.7717/peerj.453: https://doi.org/10.7717/peerj.453

    .. [5] John D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in
        Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
        doi:`10.1109/MCSE.2007.55`_

    .. _10.1109/MCSE.2007.55: https://doi.org/10.1109/MCSE.2007.55

    .. [6] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, et al., "SciPy
        1.0: Fundamental Algorithms for Scientific Computing in Python", Nature
        Methods, vol. 17, pp 261-272, 2020, doi:`10.1038/s41592-019-0686-2`_
    
    .. _10.1038/s41592-019-0686-2: https://doi.org/10.1038/s41592-019-0686-2

    .. [7] Paul Rosin, "Unimodal thresholding", Pattern recognition, vol. 34,
        no. 11, pp 2083-2096, 2001, doi:`10.1016/S0031-3203(00)00136-9`_

    .. _10.1016/S0031-3203(00)00136-9: https://doi.org/10.1016/S0031-3203(00)00136-9

    """
    cdef int idx
    cdef int idx_min
    cdef int idx_max
    cdef int idx_dist = 0
    
    cdef long long hst_min
    cdef long long hst_max
    cdef long long cross_ab
    
    cdef double sum_hist
    cdef double dist
    cdef double dist_max = 0.0

    cdef np.ndarray[np.int64_t, ndim=1] p0
    cdef np.ndarray[np.int64_t, ndim=1] p1
    cdef np.ndarray[np.int64_t, ndim=1] a
    cdef np.ndarray[np.int64_t, ndim=1] b

# Copy data    
    hist = hist.copy()

# Select maximum
    idx_max = np.argmax(hist)
    hst_max = hist[idx_max]
    
    p0 = np.array([idx_max, hst_max])

# Disregard data left from maximum
    hist[0:idx_max] = 0.

# All observations within 3 sigma (in case of normal distribution)
    sum_hist = 0.997*np.sum(hist)

# Find tail end
    for idx in range(hist.size):
        sum_hist = sum_hist - hist[idx]
        if (sum_hist > 0.):
            idx_min = idx
            hst_min = hist[idx]
        else:
            break

    p1 = np.array([idx_min, hst_min])

# Compute maximum distance and location 
    for idx in range(idx_max,idx_min):
        hst = hist[idx]

        a = p0 - p1
        b = np.array([idx, hst]) - p1
        cross_ab = a[0]*b[1]-b[0]*a[1]
        dist = abs(cross_ab)/norm(a)
        if (dist > dist_max):
            idx_dist = idx
            dist_max = dist

    return idx_dist
