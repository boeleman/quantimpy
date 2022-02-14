r"""
This module performs various morphological operations on 2D and 3D Numpy [1]_
arrays. These computations can handle both isotropic and anisotropic image
resolutions.

Notes
----------
To perform morphological operations this library uses the Euclidean distance
transform [2]_. These transforms are computed using the `MLAEDT-3D`_ library.

.. _MLAEDT-3D: https://pypi.org/project/edt/
"""

import numpy as np
import edt

###############################################################################
# {{{ erode

def erode(image, dist, res = None):
    r"""
    Morphologically erode a binary Numpy array

    This function performs the morphological erosion on the binary
    Numpy array `image`. Both 2D and 3D arrays are supported. Optionally, the
    (anisotropic) resolution of the array can be provided using the Numpy array
    `res`. When a resolution array is provided it needs to be of the same
    dimension as the 'image' array. 

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    dist : {int, float} 
        The distance away from the interface to which an array is dilated in the
        same unit of length as used in the resolution.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, bool
        This function returns a morphologically eroded Numpy array. The return
        data type is `bool`.

    See Also
    --------
    ~quantimpy.morphology.erode_map
    ~quantimpy.morphology.dilate

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphologcally dilated image can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:112,16:112] = square(96,dtype=bool)

        erosion = mp.erode(image,10)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(erosion[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:112,16:112,16:112] = cube(96,dtype=bool)

        erosion = mp.erode(image,10)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(erosion[:,:,64])
        plt.show()

    """
    if (image.dtype != "bool"):
        raise ValueError("Input image needs to be binary (data type bool)")

# Rescale resolution
    factor = 1.0
    if not (res is None):
        factor = np.amin(res)
        res = res/factor

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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(image, anisotropy=(res0, res1)) >= dist/factor
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(image, anisotropy=(res0, res1, res2)) >= dist/factor
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
# {{{ dilate

def dilate(image, dist, res = None):
    r"""
    Morphologically dilate a binary Numpy array

    This function performs the morphological dilation operation on the binary
    Numpy array `image`. Both 2D and 3D arrays are supported. Optionally, the
    (anisotropic) resolution of the array can be provided using the Numpy array
    `res`. When a resolution array is provided it needs to be of the same
    dimension as the 'image' array. 

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    dist : {int, float} 
        The distance away from the interface to which an array is dilated in the
        same unit of length as used in the resolution.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, bool
        This function returns a morphologically dilated Numpy array. The return
        data type is `bool`.

    See Also
    --------
    ~quantimpy.morphology.erode
    ~quantimpy.morphology.dilate_map

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphologcally dilated image can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:112,16:112] = square(96,dtype=bool)

        dilation = mp.dilate(image,10)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(dilation[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:112,16:112,16:112] = cube(96,dtype=bool)

        dilation = mp.dilate(image,10)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(dilation[:,:,64])
        plt.show()

    """
    if (image.dtype != "bool"):
        raise ValueError("Input image needs to be binary (data type bool)")

# Rescale resolution
    factor = 1.0
    if not (res is None):
        factor = np.amin(res)
        res = res/factor
    
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(np.logical_not(image), anisotropy=(res0, res1)) < dist/factor
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(np.logical_not(image), anisotropy=(res0, res1, res2)) < dist/factor
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
# {{{ open

def open(erosion, dist, res = None):
    r"""
    Morphologically open a binary Numpy array

    This function is an alias for the function
    :func:`~quantimpy.morphology.dilate`. Together with the
    :func:`~quantimpy.morphology.erode` function this function performs the
    morphological opening operation on the binary Numpy array `erosion`. Both
    2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of
    the array can be provided using the Numpy array `res`. When a resolution
    array is provided it needs to be of the same dimension as the 'dilation'
    array. 

    Parameters
    ----------
    erosion : ndarray, bool
        Erosion can be either a 2D or 3D array of data type `bool`.
    dist : {int, float} 
        The distance away from the interface to which an array is opened in the
        same unit of length as used in the resolution.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, bool
        This function returns a morphologically opened Numpy array. The return
        data type is `bool`.

    See Also
    --------
    ~quantimpy.morphology.dilate
    ~quantimpy.morphology.erode
    ~quantimpy.morphology.close

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphologcally opened image can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:112,16:112] = square(96,dtype=bool)

        erosion = mp.erode(image,10)
        opening = mp.open(erosion,10)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(opening[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:112,16:112,16:112] = cube(96,dtype=bool)

        erosion = mp.erode(image,10)
        opening = mp.open(erosion,10)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(opening[:,:,64])
        plt.show()

    """

    return dilate(erosion, dist, res)    

# }}}
###############################################################################
# {{{ close

def close(dilation, dist, res = None):
    r"""
    Morphologically close a binary Numpy array

    This function is an alias for the function
    :func:`~quantimpy.morphology.erode`. Together with the
    :func:`~quantimpy.morphology.dilate` function this function performs the
    morphological closing operation on the binary Numpy array `dilation`. Both
    2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of
    the array can be provided using the Numpy array `res`. When a resolution
    array is provided it needs to be of the same dimension as the 'dilation'
    array. 

    Parameters
    ----------
    dilation : ndarray, bool
        Dilation can be either a 2D or 3D array of data type `bool`.
    dist : {int, float} 
        The distance away from the interface to which an array is closed in the
        same unit of length as used in the resolution.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, bool
        This function returns a morphologically closed Numpy array. The return
        data type is `bool`.

    See Also
    --------
    ~quantimpy.morphology.erode
    ~quantimpy.morphology.dilate
    ~quantimpy.morphology.open

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphologcally closed image can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([127,128],dtype=bool)
        image[17:65,16:64] = square(48,dtype=bool)
        image[65:113,64:112] = square(48,dtype=bool)

        dilation = mp.dilate(image,10)
        closing  = mp.close(dilation,10)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(closing[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:64,16:64,16:64] = cube(48,dtype=bool)
        image[64:112,64:112,64:112] = cube(48,dtype=bool)

        dilation = mp.dilate(image,10)
        closing  = mp.close(dilation,10)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(closing[:,:,64])
        plt.show()

    """

    return erode(dilation, dist, res)    

# }}}
###############################################################################
# {{{ erode_map

def erode_map(image, res = None):
    r"""
    Compute a morphological erosion map of a Numpy array

    This function computes a morphological erosion map of the binary Numpy array
    `image`. Both 2D and 3D arrays are supported. Optionally, the (anisotropic)
    resolution of the array can be provided using the Numpy array `res`. When a
    resolution array is provided it needs to be of the same dimension as the
    'image' array. 

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in
        all directions. If a resolution is provided it needs to be of the same
        dimension as the dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, uint16
        This function returns an erosion map of a binary Numpy array. The return
        data type is `uint16`.

    See Also
    --------
    ~quantimpy.morphology.erode
    ~quantimpy.morphology.open_map

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphological erosion map can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (disk)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        erosionMap = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(erosionMap[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (ball)
        from quantimpy import morphology as mp

        image = np.zeros([129,127,128],dtype=bool)
        image[15:112,14:111,16:113] = ball(48,dtype=bool)

        erosionMap = mp.erode_map(image)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(erosionMap[:,:,64])
        plt.show()

    """
    if (image.dtype != "bool"):
        raise ValueError("Input image needs to be binary (data type bool)")

# Rescale resolution
    if not (res is None):
        factor = np.amin(res)
        res = res/factor

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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(image, anisotropy=(res0, res1)).astype(np.uint16)
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return edt.edt(image, anisotropy=(res0, res1, res2)).astype(np.uint16)
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
# {{{ dilate_map

def dilate_map(image, res = None):
    r"""
    Compute a morphological dilation map of a Numpy array

    This function computes a morphological dilation map of the binary Numpy array
    `image`. Both 2D and 3D arrays are supported. Optionally, the (anisotropic)
    resolution of the array can be provided using the Numpy array `res`. When a
    resolution array is provided it needs to be of the same dimension as the
    'dilation' array. 

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, uint16
        This function returns a dilation map of a binary Numpy array. The return
        data type is `uint16`.

    See Also
    --------
    ~quantimpy.morphology.erode_map
    ~quantimpy.morphology.close_map

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphological closing map can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (disk)
        from quantimpy import morphology as mp

        image = np.zeros([127,128],dtype=bool)
        image[16:113,16:113] = disk(48,dtype=bool)

        dilationMap = mp.dilate_map(image)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(dilationMap[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (ball)
        from quantimpy import morphology as mp

        image = np.zeros([129,127,128],dtype=bool)
        image[15:112,14:111,16:113] = ball(48,dtype=bool)

        dilationMap = mp.dilate_map(image)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(dilationMap[:,:,64])
        plt.show()
    
    """
    if (image.dtype != "bool"):
        raise ValueError("Input image needs to be binary (data type bool)")

# Rescale resolution
    if not (res is None):
        factor = np.amin(res)
        res = res/factor
    
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return (np.ones_like(image)*np.iinfo(np.uint16).max - edt.edt(np.logical_not(image), anisotropy=(res0, res1)).astype(np.uint16)).astype(np.uint16)
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
            raise ValueError("Input image and resolution need to be the same dimension")

        return (np.ones_like(image)*np.iinfo(np.uint16).max - edt.edt(np.logical_not(image), anisotropy=(res0, res1, res2)).astype(np.uint16)).astype(np.uint16)
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
# {{{ open_map

def open_map(erosion_map, res = None):

    r"""
    Compute a morphological opening map of a Numpy array

    Together with the :func:`~quantimpy.morphology.erode_map` function this
    function computes a morphological opening map of the Numpy array
    `erosion_map`. Both 2D and 3D arrays are supported. Optionally, the
    (anisotropic) resolution of the array can be provided using the Numpy array
    `res`. When a resolution array is provided it needs to be of the same
    dimension as the 'dilation' array. 

    Parameters
    ----------
    erosion_map : ndarray, float
        Erosion_map can be either a 2D or 3D array of data type `float`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, uint16
        This function returns a distance map of a morphologically closed Numpy array. The return
        data type is `uint16`.

    See Also
    --------
    ~quantimpy.morphology.dilate_map
    ~quantimpy.morphology.open_map

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphological closing map can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:112,16:112] = square(96,dtype=bool)

        erosionMap = mp.erode_map(image)
        openingMap = mp.open_map(erosionMap)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(openingMap[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:112,16:112,16:112] = cube(96,dtype=bool)

        erosionMap = mp.erode_map(image)
        openingMap = mp.open_map(erosionMap)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(openingMap[:,:,64])
        plt.show()

    References
    ----------
    .. [1] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt et al.,
        "Array programming with NumPy", Nature, vol. 585, pp 357-362, 2020,
        doi:`10.1038/s41586-020-2649-2`_

    .. _10.1038/s41586-020-2649-2: https://doi.org/10.1038/s41586-020-2649-2

    .. [2] Ingemar Ragnemalm, "Fast erosion and dilation by contour processing and
        thresholding of distance maps", Pattern recognition letters, vol. 13, no. 3,
        pp 161-166, 1992, doi:`10.1016/0167-8655(92)90055-5`_

    .. _10.1016/0167-8655(92)90055-5: https://doi.org/10.1016/0167-8655(92)90055-5

    .. [3] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
        François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart,
        Tony Yu and the scikit-image contributors. "scikit-image: Image
        processing in Python." PeerJ 2:e453 (2014) doi: `10.7717/peerj.453`_

    .. _10.7717/peerj.453: https://doi.org/10.7717/peerj.453

    .. [4] John D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in
        Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
        doi:`10.1109/MCSE.2007.55`_

    .. _10.1109/MCSE.2007.55: https://doi.org/10.1109/MCSE.2007.55
    """
    if (erosion_map.dtype != "uint16"):
        raise ValueError("Input image needs to be data type uint16")

# Rescale resolution
    if not (res is None):
        factor = np.amin(res)
        res = res/factor
    
    if (erosion_map.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.size == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError("Input image and resolution need to be the same dimension")

        open_map = np.zeros_like(erosion_map).astype(np.uint16)

        for i in range(np.max(erosion_map)+1):
            print("Open map step: ", i)

            dilation = np.logical_not(erosion_map >= i)
            
            dilation = edt.edt(dilation, anisotropy=(res0, res1)).astype(np.uint16)
            dilation = dilation < i
            
            open_map[dilation] = i

        return open_map
    elif (erosion_map.ndim == 3):
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
            raise ValueError("Input image and resolution need to be the same dimension")

        open_map = np.zeros_like(erosion_map).astype(np.uint16)

        for i in range(np.max(erosion_map)+1):
            print("Open map step: ", i)

            dilation = np.logical_not(erosion_map >= i)
            
            dilation = edt.edt(dilation, anisotropy=(res0, res1, res2)).astype(np.uint16)
            dilation = dilation < i
            
            open_map[dilation] = i

        return open_map
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
# {{{ close_map

def close_map(dilation_map, res = None):
    r"""
    Compute a morphological closing map of a Numpy array

    Together with the :func:`~quantimpy.morphology.dilate_map` function this
    function computes a morphological closing map of the Numpy array
    `dilation_map`. Both 2D and 3D arrays are supported. Optionally, the
    (anisotropic) resolution of the array can be provided using the Numpy array
    `res`. When a resolution array is provided it needs to be of the same
    dimension as the 'dilation' array. 

    Parameters
    ----------
    dilation_map : ndarray, float
        Dilation_map can be either a 2D or 3D array of data type `float`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        dilation array and all elements of the resolution array need to be
        larger than or equal to one.

    Returns
    -------
    out : ndarray, uint16
        This function returns a distance map of a morphologically closed Numpy array. The return
        data type is `uint16`.

    See Also
    --------
    ~quantimpy.morphology.dilate_map
    ~quantimpy.morphology.open_map

    Examples
    --------
    These examples use the skimage Python package [3]_ and the Matplotlib Python
    package [4]_. For a 2D image a morphological closing map can be computed
    using the following example:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (square)
        from quantimpy import morphology as mp

        image = np.zeros([128,128],dtype=bool)
        image[16:64,16:64] = square(48,dtype=bool)
        image[64:112,64:112] = square(48,dtype=bool)

        dilationMap = mp.dilate_map(image)
        closingMap = mp.close_map(dilationMap)

        plt.gray()
        plt.imshow(image[:,:])
        plt.show()

        plt.gray()
        plt.imshow(closingMap[:,:])
        plt.show()

    For a 3D image the following example can be used: 

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.morphology import (cube)
        from quantimpy import morphology as mp

        image = np.zeros([128,128,128],dtype=bool)
        image[16:64,16:64,16:64] = cube(48,dtype=bool)
        image[64:112,64:112,64:112] = cube(48,dtype=bool)

        dilationMap = mp.dilate_map(image)
        closingMap = mp.close_map(dilationMap)

        plt.gray()
        plt.imshow(image[:,:,64])
        plt.show()

        plt.gray()
        plt.imshow(closingMap[:,:,64])
        plt.show()

    """
    if (dilation_map.dtype != "uint16"):
        raise ValueError("Input image needs to be data type uint16")

# Rescale resolution
    if not (res is None):
        factor = np.amin(res)
        res = res/factor
    
    if (dilation_map.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        elif (res.size == 2):
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
        else:
            raise ValueError("Input image and resolution need to be the same dimension")

        close_map = np.ones_like(dilation_map).astype(np.uint16)*np.iinfo(np.uint16).max
        dilation_map = close_map - dilation_map

        for i in range(np.max(dilation_map)+1):
            print("Close map step: ", i)

            erosion = dilation_map < i
            
            erosion = edt.edt(erosion, anisotropy=(res0, res1)).astype(np.uint16)
            erosion = erosion >= i
            
            close_map[~erosion] = np.iinfo(np.uint16).max - i

        return close_map
    elif (dilation_map.ndim == 3):
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
            raise ValueError("Input image and resolution need to be the same dimension")

        close_map = np.ones_like(dilation_map).astype(np.uint16)*np.iinfo(np.uint16).max
        dilation_map = close_map - dilation_map

        for i in range(np.max(dilation_map)+1):
            print("Close map step: ", i)

            erosion = dilation_map < i
            
            erosion = edt.edt(erosion, anisotropy=(res0, res1, res2)).astype(np.uint16)
            erosion = erosion >= i
            
            close_map[~erosion] = np.iinfo(np.uint16).max - i

        return close_map
    else:
        raise ValueError("Only 2D and 3D images are supported")

# }}}
###############################################################################
