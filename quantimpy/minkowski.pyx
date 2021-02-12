r"""Compute the Minkowski functionals and functions

This module can compute both the Minkowski functionals and functions for 2D and
3D Numpy arrays. These computations can handle both isotropic and anisotropic
image resolutions.

Notes
----------
More information about the used algorithm can be found in the book "Statistical
analysis of microstructures in materials science" by Joachim Ohser and Frank
Mücklich [1]_.

References
----------
.. [1] Joachim Ohser and Frank Mücklich, "Statistical analysis of
    microstructures in materials science", Wiley and Sons, New York, 2000, ISBN:
    0471974862

"""

import numpy as np
cimport numpy as np

np.import_array()

###############################################################################
# {{{ functionals

cpdef functionals(np.ndarray image, res = None):
    r"""Compute the Minkowski functionals in 2D or 3D.

    This function computes the Minkowski functionals for Numpy array `image`. Both
    2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of the
    array can be provided using the Numpy array `res`. When a resolution array is
    provided it needs to be the same dimension as the image array.

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1mm/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        image array.

    Returns
    -------
    out : tuple, float
        In the case of a 2D image this function returns a tuple of the area,
        length, and euler densities. In the case of a 3D image this function
        returns a tuple of the volume, surface, curvature, and euler densities.
        The return data type is `float`.


    See Also
    --------
    quantimpy.minkowski.functions_open
    quantimpy.minkowski.functions_close

    Notes
    -----

    The definition of the Minkowski functionals follows the convention in the
    physics literature [2]_.

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
    These examples use the skimage Python package [3]_. For a 2D image the
    Minkowski fucntionals can be computed using the following example:

    .. code-block:: python

        import numpy as np    
        from quantimpy import minkowski as mk
        from skimage.morphology import (disk)

        image = np.zeros([128,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        minkowski = mk.functionals(image)

        # Compute Minkowski functionals for image with anisotropic resolution
        res = np.array([2, 1])
        minkowski = mk.functionals(image,res)

    For a 3D image the Minkowski fucntionals can be computed using the following
    example:

    .. code-block:: python

        import numpy as np    
        from quantimpy import minkowski as mk
        from skimage.morphology import (ball)

        image = np.zeros([128,128,128],dtype=bool)
        image[16:113,16:113,16:113] = ball(48,dtype=bool)

        minkowski = mk.functionals(image)

        # Compute Minkowski functionals for image with anisotropic resolution
        res = np.array([2, 1, 3])
        minkowski = mk.functionals(image,res)

    References
    ----------
    .. [2] Klaus R. Mecke, "Additivity, convexity, and beyond: applications of
        Minkowski Functionals in statistical physics" in "Statistical Physics
        and Spatial Statistics", pp 111–184, Springer (2000) doi:
        `10.1007/3-540-45043-2_6`_

    .. _10.1007/3-540-45043-2_6: https://doi.org/10.1007/3-540-45043-2_6

    .. [3] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
        François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart,
        Tony Yu and the scikit-image contributors. "scikit-image: Image
        processing in Python." PeerJ 2:e453 (2014) doi: `10.7717/peerj.453`_

    .. _10.7717/peerj.453: https://doi.org/10.7717/peerj.453


    """

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

        return _functionals2D(image, res0, res1)
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

        return _functionals3D(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "minkowskic.h":
    bint cFunctionals2D(
        unsigned short* image, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1, 
        double* area, 
        double* length, 
        double* euler4, 
        double* euler8)


def _functionals2D(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        double res0, 
        double res1):

    cdef double area 
    cdef double length 
    cdef double euler4 
    cdef double euler8
    
    image = np.ascontiguousarray(image)
    
    status = cFunctionals2D(
        &image[0,0], 
        image.shape[0], image.shape[1],
        res0, res1,
        &area, &length, &euler4, &euler8)

    assert status == 0
    return area, length, euler8


cdef extern from "minkowskic.h":
    bint cFunctionals3D(
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


def _functionals3D(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        double res0, 
        double res1, 
        double res2):

    cdef double volume 
    cdef double surface 
    cdef double curvature 
    cdef double euler6
    cdef double euler26
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(
        image,dtype=np.uint16)
    
    status = cFunctionals3D(
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
    return volume, surface, curvature, euler26

# }}}
###############################################################################
# {{{ functions_open

cpdef functions_open(np.ndarray closing, res = None):

    if not (closing.dtype == 'uint16'):
        closing = closing.astype('uint16')
    
    if (closing.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.ndim == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _FunctionsOpen2D(closing, res0, res1)
    elif (closing.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        elif (res.ndim == 3):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _FunctionsOpen3D(closing, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D closings')


cdef extern from "minkowskic.h":
    bint cFunctionsOpen2D(
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


def _FunctionsOpen2D(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] closing, 
        res0, res1):
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

    status = cFunctionsOpen2D(
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
    return dist, area, length, euler8


cdef extern from "minkowskic.h":
    bint cFunctionsOpen3D(
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


def _FunctionsOpen3D(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] closing, 
        res0, 
        res1, 
        res2):

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

    status = cFunctionsOpen3D(
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
    return dist, volume, surface, curvature, euler26

# }}}
###############################################################################
# {{{ functions_close

cpdef functions_close(np.ndarray closing, res = None):

    if not (closing.dtype == 'uint16'):
        closing = closing.astype('uint16')
    
    if (closing.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.ndim == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _FunctionsClose2D(closing, res0, res1)
    elif (closing.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        elif (res.ndim == 3):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]
        else:
            raise ValueError('Input image and resolution need to be the same dimension')

        return _FunctionsClose3D(closing, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D closings')


cdef extern from "minkowskic.h":
    bint cFunctionsClose2D(
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


def _FunctionsClose2D(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] closing, 
        res0, 
        res1):

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

    status = cFunctionsClose2D(
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
    return dist, area, length, euler8


cdef extern from "minkowskic.h":
    bint cFunctionsClose3D(
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


def _FunctionsClose3D(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] closing, 
        res0, 
        res1, 
        res2):

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

    status = cFunctionsClose3D(
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
    return dist, volume, surface, curvature, euler26

# }}}
###############################################################################
