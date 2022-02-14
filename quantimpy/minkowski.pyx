r"""Compute the Minkowski functionals and functions

This module can compute both the Minkowski functionals and functions for 2D and
3D Numpy [1]_ arrays. These computations can handle both isotropic and anisotropic
image resolutions.

Notes
----------
More information about the used algorithm can be found in the book "Statistical
analysis of microstructures in materials science" by Joachim Ohser and Frank
Mücklich [2]_.
"""

import cython
import numpy as np
cimport numpy as np

np.import_array()

###############################################################################
# {{{ functionals

@cython.binding(True)
cpdef functionals(np.ndarray image, res = None, norm=False):
    r"""Compute the Minkowski functionals in 2D or 3D.

    This function computes the Minkowski functionals for the Numpy array `image`. Both
    2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of the
    array can be provided using the Numpy array `res`. When a resolution array is
    provided it needs to be of the same dimension as the image array.

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        image array.
    norm : bool, defaults to False
        When norm=True the functionals are normalized with the total area or
        volume of the image. Defaults to norm=False.

    Returns
    -------
    out : tuple, float
        In the case of a 2D image this function returns a tuple of the area,
        length, and the Euler characteristic. In the case of a 3D image this
        function returns a tuple of the volume, surface, curvature, and the
        Euler characteristic. The return data type is `float`.

    See Also
    --------
    ~quantimpy.minkowski.functions_open
    ~quantimpy.minkowski.functions_close

    Notes
    -----

    The definition of the Minkowski functionals follows the convention in the
    physics literature [3]_.

    Considering a 2D body, :math:`X`, with a smooth boundary, :math:`\delta X`,
    the following functionals are computed:

    .. math:: M_{0} (X) &= \int_{X} d s, \\
              M_{1} (X) &= \frac{1}{2 \pi} \int_{\delta X} d c, \text{ and } \\
              M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X} \left[\frac{1}{R} \right] d c,

    where :math:`d s` is a surface element and :math:`d c` is a circumference
    element. :math:`R` is the radius of the local curvature. This results in the
    following definitions for the surface area, :math:`S = M_{0} (X)`,
    circumference, :math:`C = 2 \pi M_{1} (X)`, and the 2D Euler characteristic,
    :math:`\chi (X) = \pi M_{2} (X)`. 

    Considering a 3D body, :math:`X`, with a smooth boundary surface, :math:`\delta
    X`, the following functionals are computed:

    .. math:: M_{0} (X) &= V = \int_{X} d v, \\
              M_{1} (X) &= \frac{1}{8} \int_{\delta X} d s, \\
              M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X}  \frac{1}{2} \left[\frac{1}{R_{1}} + \frac{1}{R_{2}}\right] d s, \text{ and } \\
              M_{3} (X) &= \frac{3}{(4 \pi)^{2}} \int_{\delta X} \left[\frac{1}{R_{1} R_{2}}\right] d s,

    where :math:`d v` is a volume element and :math:`d s` is a surface element.
    :math:`R_{1}` and :math:`R_{2}` are the principal radii of curvature of
    surface element :math:`d s`. This results in the following definitions for
    the volume, :math:`V = M_{0} (X)`, surface area, :math:`S = 8 M_{1} (X)`,
    integral mean curvature, :math:`H = 2 \pi^{2} M_{2} (X)`, and the 3D Euler
    characteristic, :math:`\chi (X) = 4 \pi/3 M_{3} (X)`.

    Examples
    --------
    These examples use the scikit-image Python package [4]_ and the Matplotlib Python
    package [5]_. For a 2D image the Minkowski functionals can be computed using
    the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (disk)
        from quantimpy import minkowski as mk

        image = np.zeros([128,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        minkowski = mk.functionals(image)
        print(minkowski)

        # Compute Minkowski functionals for image with anisotropic resolution
        res = np.array([2, 1])
        minkowski = mk.functionals(image,res)
        print(minkowski)

    For a 3D image the Minkowski functionals can be computed using the following
    example:

    .. code-block:: python

        import numpy as np    
        import matplotlib.pyplot as plt
        from skimage.morphology import (ball)
        from quantimpy import minkowski as mk

        image = np.zeros([128,128,128],dtype=bool)
        image[16:113,16:113,16:113] = ball(48,dtype=bool)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        minkowski = mk.functionals(image)
        print(minkowski)

        # Compute Minkowski functionals for image with anisotropic resolution
        res = np.array([2, 1, 3])
        minkowski = mk.functionals(image,res)
        print(minkowski)

    """
# Decompose resolution in number larger than one and a pre-factor
    factor = 1.0
    if not (res is None):
        factor = np.amin(res)
        res = res/factor

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')

    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.size == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functionals_2d(image, res0, res1, factor, norm)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        elif (res.size == 3):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functionals_3d(image, res0, res1, res2, factor, norm)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "minkowskic.h":
    bint c_functionals_2d(
        unsigned short* image, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1, 
        double* area, 
        double* length, 
        double* euler4, 
        double* euler8)


def _functionals_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        double res0, 
        double res1,
        double factor,
        bint norm):

    cdef double area 
    cdef double length 
    cdef double euler4 
    cdef double euler8
    
    image = np.ascontiguousarray(image)
    
    status = c_functionals_2d(
        &image[0,0], 
        image.shape[0], image.shape[1],
        res0, res1,
        &area, &length, &euler4, &euler8)

    assert status == 0
    if norm:
        total_area = image.shape[0]*image.shape[1]*res0*res1
        return area/total_area, length/(total_area*factor), euler8/(total_area*factor**2)
    else:
        return area*factor**2, length*factor, euler8


cdef extern from "minkowskic.h":
    bint c_functionals_3d(
        unsigned short* image, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2, 
        double* volume, 
        double* surface, 
        double* curvature, 
        double* euler6, 
        double* euler26)


def _functionals_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        double res0, 
        double res1, 
        double res2,
        double factor,
        bint norm):

    cdef double volume 
    cdef double surface 
    cdef double curvature 
    cdef double euler6
    cdef double euler26
   
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_functionals_3d(
        &image[0,0,0],
        image.shape[0], 
        image.shape[1], 
        image.shape[2],
        res0, 
        res1, 
        res2,
        &volume, 
        &surface, 
        &curvature, 
        &euler6, 
        &euler26)

    assert status == 0
    if norm:
        total_volume = image.shape[0]*image.shape[1]*image.shape[2]*res0*res1*res2
        return volume/(total_volume), surface/(total_volume*factor), curvature/(total_volume*factor**2), euler26/(total_volume*factor**3)
    else:
        return volume*factor**3, surface*factor**2, curvature*factor, euler26

# }}}
###############################################################################
# {{{ functions_open

@cython.binding(True)
cpdef functions_open(np.ndarray opening, res = None, norm=False):
    r"""
    Compute the Minkowski functions in 2D or 3D.

    This function computes the Minkowski functionals as function of the
    grayscale values in the Numpy array `opening`. Both 2D and 3D arrays are supported.
    Optionally, the (anisotropic) resolution of the array can be provided using
    the Numpy array `res`. When a resolution array is provided it needs to be of
    the same dimension as the 'opening' array. 

    The algorithm iterates over all grayscale values present in the array,
    starting at the smallest value (black). For every grayscale value the array
    is converted into a binary image where values larger than the grayscale
    value become one (white) and all other values become zero (black). For each
    of these binary images the minkowski functionals are computed according to
    the function :func:`~quantimpy.minkowski.functionals`.

    This function can be used in combination with the
    :func:`~quantimpy.morphology` module to compute the Minkowski functions of
    different morphological distance maps.

    Parameters
    ----------
    opening : ndarray, float
        Opening can be either a 2D or 3D array of data type `float`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        image array.
    norm : bool, defaults to False
        When norm=True the functions are normalized with the total area or
        volume of the image. Defaults to norm=False.

    Returns
    -------
    out : tuple, ndarray, float
        In the case of a 2D image this function returns a tuple of Numpy arrays
        consisting of the distance (assuming one grayscale value is used per
        unit of length), area, length, and the Euler characteristic. In the
        case of a 3D image this function returns a tuple of Numpy arrays
        consistenting of the distance, volume, surface, curvature, and the Euler
        characteristic. The return data type is `float`.

    See Also
    --------
    ~quantimpy.minkowski.functionals
    ~quantimpy.minkowski.functions_close
    ~quantimpy.morphology

    Examples
    --------
    These examples use the Skimage Python package [4]_ and the Matplotlib Python
    package [5]_. For a 2D image the Minkowski functions can be computed using
    the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (disk)
        from quantimpy import morphology as mp
        from quantimpy import minkowski as mk

        image = np.zeros([128,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        erosion_map = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(erosion_map[:,:])
        plt.show()

        dist, area, length, euler = mk.functions_open(erosion_map)

        plt.plot(dist,area)
        plt.show()

        plt.plot(dist,length)
        plt.show()

        plt.plot(dist,euler)
        plt.show()

    For a 3D image the Minkowski functionals can be computed using the following
    example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (ball)
        from quantimpy import morphology as mp
        from quantimpy import minkowski as mk

        image = np.zeros([128,128,128],dtype=bool)
        image[16:113,16:113,16:113] = ball(48,dtype=bool)

        erosion_map = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(erosion_map[:,:,64])
        plt.show()

        dist, volume, surface, curvature, euler = mk.functions_open(erosion_map)

        plt.plot(dist,volume)
        plt.show()

        plt.plot(dist,surface)
        plt.show()

        plt.plot(dist,curvature)
        plt.show()

        plt.plot(dist,euler)
        plt.show()

    References
    ----------
    .. [1] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt et al.,
        "Array programming with NumPy", Nature, vol. 585, pp 357-362, 2020,
        doi:`10.1038/s41586-020-2649-2`_

    .. _10.1038/s41586-020-2649-2: https://doi.org/10.1038/s41586-020-2649-2

    .. [2] Joachim Ohser and Frank Mücklich, "Statistical analysis of
        microstructures in materials science", Wiley and Sons, New York (2000) ISBN:
        0471974862

    .. [3] Klaus R. Mecke, "Additivity, convexity, and beyond: applications of
        Minkowski Functionals in statistical physics" in "Statistical Physics
        and Spatial Statistics", pp 111–184, Springer (2000) doi:
        `10.1007/3-540-45043-2_6`_

    .. _10.1007/3-540-45043-2_6: https://doi.org/10.1007/3-540-45043-2_6

    .. [4] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
        François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart,
        Tony Yu and the scikit-image contributors. "scikit-image: Image
        processing in Python." PeerJ 2:e453 (2014) doi: `10.7717/peerj.453`_

    .. _10.7717/peerj.453: https://doi.org/10.7717/peerj.453

    .. [5] John D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in
        Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
        doi:`10.1109/MCSE.2007.55`_

    .. _10.1109/MCSE.2007.55: https://doi.org/10.1109/MCSE.2007.55
    """
# Decompose resolution in number larger than one and a pre-factor
    factor = 1.0        
    if not (res is None):
        factor = np.amin(res)
        res = res/factor
    
    if not (opening.dtype == 'uint16'):
        opening = opening.astype('uint16')
    
    if (opening.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.size == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functions_open_2d(opening, res0, res1, factor, norm)
    elif (opening.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        elif (res.size == 3):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functions_open_3d(opening, res0, res1, res2, factor, norm)
    else:
        raise ValueError('Can only handle 2D or 3D openings')


cdef extern from "minkowskic.h":
    bint c_functions_open_2d(
        unsigned short* opening, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1, 
        double* dist, 
        double* area, 
        double* length, 
        double* euler4, 
        double* euler8)


def _functions_open_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] opening, 
        double res0, 
        double res1, 
        double factor,
        bint norm):

    cdef int dim3

    opening = np.ascontiguousarray(opening)

    dim3 = np.amax(opening) - np.amin(opening)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] area = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] length = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler4 = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler8 = np.empty(
        dim3,dtype=np.double)

    status = c_functions_open_2d(
        &opening[0,0], 
        opening.shape[0], 
        opening.shape[1],
        res0, 
        res1,
        &dist[0], 
        &area[0],
        &length[0], 
        &euler4[0], 
        &euler8[0])

    assert status == 0
    if norm:
        total_area = opening.shape[0]*opening.shape[1]*res0*res1
        return dist*factor, area/(total_area), length/(total_area*factor), euler8/(total_area*factor**2)
    else:
        return dist*factor, area*factor**2, length*factor, euler8


cdef extern from "minkowskic.h":
    bint c_functions_open_3d(
        unsigned short* opening, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2, 
        double* dist, 
        double* volume, 
        double* surface, 
        double* curvature, 
        double* euler6, 
        double* euler26)


def _functions_open_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] opening, 
        double res0, 
        double res1, 
        double res2,
        double factor,
        bint norm):

    cdef int dim3

    opening = np.ascontiguousarray(opening)

    dim3 = np.amax(opening) - np.amin(opening)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] volume = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] surface = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] curvature = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler6 = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler26 = np.empty(
        dim3,dtype=np.double)

    status = c_functions_open_3d(
        &opening[0,0,0], 
        opening.shape[0],
        opening.shape[1],
        opening.shape[2],
        res0,
        res1,
        res2,
        &dist[0],
        &volume[0],
        &surface[0],
        &curvature[0],
        &euler6[0],
        &euler26[0])

    assert status == 0
    if norm:
        total_volume = opening.shape[0]*opening.shape[1]*opening.shape[2]*res0*res1*res2
        return dist*factor, volume/(total_volume), surface/(total_volume*factor), curvature/(total_volume*factor**2), euler26/(total_volume*factor**3)
    else:
        return dist*factor, volume*factor**3, surface*factor**2, curvature*factor, euler26


# }}}
###############################################################################
# {{{ functions_close

@cython.binding(True)
cpdef functions_close(np.ndarray closing, res = None, norm=False):
    r"""
    Compute the Minkowski functions in 2D or 3D.

    This function computes the Minkowski functionals as function of the
    grayscale values in the Numpy array `closing`. Both 2D and 3D arrays are supported.
    Optionally, the (anisotropic) resolution of the array can be provided using
    the Numpy array `res`. When a resolution array is provided it needs to be of
    the same dimension as the 'closing' array. 

    The algorithm iterates over all grayscale values present in the array,
    starting at the largest value (white). For every grayscale value the array
    is converted into a binary image where values larger than the grayscale
    value become one (white) and all other values become zero (black). For each
    of these binary images the minkowski functionals are computed according to
    the function :func:`~quantimpy.minkowski.functionals`.

    This function can be used in combination with the
    :func:`~quantimpy.morphology` module to compute the Minkowski functions of
    different morphological distance maps.

    Parameters
    ----------
    closing : ndarray, float
        Closing can be either a 2D or 3D array of data type `float`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        image array.
    norm : bool, defaults to False
        When norm=True the functions are normalized with the total area or
        volume of the image. Defaults to norm=False.

    Returns
    -------
    out : tuple, ndarray, float
        In the case of a 2D image this function returns a tuple of Numpy arrays
        consisting of the distance (assuming one grayscale value is used per
        unit of length), area, length, and the Euler characteristic. In the
        case of a 3D image this function returns a tuple of Numpy arrays
        consistenting of the distance, volume, surface, curvature, and the Euler
        characteristic. The return data type is `float`.

    See Also
    --------
    ~quantimpy.minkowski.functionals
    ~quantimpy.minkowski.functions_open
    ~quantimpy.morphology

    Examples
    --------
    These examples use the scikit-image Python package [4]_ and the Matplotlib Python
    package [5]_. For a 2D image the Minkowski functions can be computed using
    the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (disk)
        from quantimpy import morphology as mp
        from quantimpy import minkowski as mk

        image = np.zeros([128,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        erosion_map = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(erosion_map[:,:])
        plt.show()

        dist, area, length, euler = mk.functions_close(erosion_map)

        plt.plot(dist,area)
        plt.show()

        plt.plot(dist,length)
        plt.show()

        plt.plot(dist,euler)
        plt.show()

    For a 3D image the Minkowski functionals can be computed using the following
    example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (ball)
        from quantimpy import morphology as mp
        from quantimpy import minkowski as mk

        image = np.zeros([128,128,128],dtype=bool)
        image[16:113,16:113,16:113] = ball(48,dtype=bool)

        erosion_map = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(erosion_map[:,:,64])
        plt.show()

        dist, volume, surface, curvature, euler = mk.functions_close(erosion_map)

        plt.plot(dist,volume)
        plt.show()

        plt.plot(dist,surface)
        plt.show()

        plt.plot(dist,curvature)
        plt.show()

        plt.plot(dist,euler)
        plt.show()

    """
# Decompose resolution in number larger than one and a pre-factor
    factor = 1.0
    if not (res is None):
        factor = np.amin(res)
        res = res/factor

    if not (closing.dtype == 'uint16'):
        closing = closing.astype('uint16')
    
    if (closing.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.size == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functions_close_2d(closing, res0, res1, factor, norm)
    elif (closing.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        elif (res.size == 3):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _functions_close_3d(closing, res0, res1, res2, factor, norm)
    else:
        raise ValueError('Can only handle 2D or 3D closings')


cdef extern from "minkowskic.h":
    bint c_functions_close_2d(
        unsigned short* closing, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1, 
        double* dist, 
        double* area, 
        double* length, 
        double* euler4, 
        double* euler8)


def _functions_close_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] closing, 
        double res0, 
        double res1,
        double factor,
        bint norm):

    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] area = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] length = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler4 = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler8 = np.empty(
        dim3,dtype=np.double)

    status = c_functions_close_2d(
        &closing[0,0], 
        closing.shape[0],
        closing.shape[1],
        res0,
        res1,
        &dist[0],
        &area[0],
        &length[0],
        &euler4[0],
        &euler8[0])

    assert status == 0
    if norm:
        total_area = closing.shape[0]*closing.shape[1]*res0*res1
        return dist*factor, area/(total_area), length/(total_area*factor), euler8/(total_area*factor**2)
    else:
        return dist*factor, area*factor**2, length*factor, euler8


cdef extern from "minkowskic.h":
    bint c_functions_close_3d(
        unsigned short* closing, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2, 
        double* dist, 
        double* volume, 
        double* surface, 
        double* curvature, 
        double* euler6, 
        double* euler26)


def _functions_close_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] closing, 
        double res0, 
        double res1, 
        double res2,
        double factor,
        bint norm):

    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] volume = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] surface = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] curvature = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler6 = np.empty(
        dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler26 = np.empty(
        dim3,dtype=np.double)

    status = c_functions_close_3d(
        &closing[0,0,0], 
        closing.shape[0],
        closing.shape[1],
        closing.shape[2],
        res0,
        res1,
        res2,
        &dist[0],
        &volume[0],
        &surface[0],
        &curvature[0],
        &euler6[0],
        &euler26[0])

    assert status == 0
    if norm:
        total_volume = closing.shape[0]*closing.shape[1]*closing.shape[2]*res0*res1*res2
        return dist*factor, volume/(total_volume), surface/(total_volume*factor), curvature/(total_volume*factor**2), euler26/(total_volume*factor**3)
    else:
        return dist*factor, volume*factor**3, surface*factor**2, curvature*factor, euler26


# }}}
###############################################################################
