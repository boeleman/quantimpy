"""
Test
"""

import numpy as np
cimport numpy as np

np.import_array()

###############################################################################
# {{{ erode

cpdef erode(np.ndarray image, int dist, res = None):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')
    
    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _erode_2d(image, dist, res0, res1)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _erode_3d(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "morphologyc.h":
    bint c_erode_2d(
        unsigned short* image, 
        unsigned short* erosion, 
        int dim0, 
        int dim1, 
        int dist, 
        double res0, 
        double res1)


def _erode_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        int dist, 
        double res0, 
        double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] erosion = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_erode_2d(
        &image[0,0],
        &erosion[0,0],
        image.shape[0],
        image.shape[1],
        dist,
        res0,
        res1,
    )

    assert status == 0
    return erosion.astype(np.bool) 


cdef extern from "morphologyc.h":
    bint c_erode_3d(
        unsigned short* image, 
        unsigned short* erosion, 
        int dim0, 
        int dim1, 
        int dim2, 
        int dist, 
        double res0, 
        double res1, 
        double res2)


def _erode_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        int dist, 
        double res0, 
        double res1, 
        double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_erode_3d(
        &image[0,0,0],
        &erosion[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        dist,
        res0,
        res1,
        res2)

    assert status == 0
    return erosion.astype(np.bool) 

# }}}
###############################################################################
# {{{ dilate

cpdef dilate(np.ndarray image, int dist, res = None):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')
    
    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _dilate_2d(image, dist, res0, res1)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _dilate_3d(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "morphologyc.h":
    bint c_dilate_2d(
        unsigned short* image, 
        unsigned short* dilation, 
        int dim0, 
        int dim1, 
        int dist, 
        double res0, 
        double res1)


def _dilate_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        int dist, 
        double res0, 
        double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] dilation = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_dilate_2d(
        &image[0,0],
        &dilation[0,0],
        image.shape[0],
        image.shape[1],
        dist,
        res0,
        res1)

    assert status == 0
    return dilation.astype(np.bool) 


cdef extern from "morphologyc.h":
    bint c_dilate_3d(
        unsigned short* image, 
        unsigned short* dilation, 
        int dim0, 
        int dim1, 
        int dim2, 
        int dist, 
        double res0, 
        double res1, 
        double res2)


def _dilate_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        int dist, 
        double res0, 
        double res1, 
        double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] dilation = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_dilate_3d(
        &image[0,0,0],
        &dilation[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        dist,
        res0,
        res1,
        res2)

    assert status == 0
    return dilation.astype(np.bool) 

# }}}
###############################################################################
# {{{ open

cpdef open(np.ndarray erosion, int dist, res = None):

    if (erosion.dtype == 'bool'):
        erosion = erosion.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input images need to be binary (data type bool)')
    
    if (erosion.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _dilate_2d(erosion, dist, res0, res1)
    elif (erosion.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _dilate_3d(erosion, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
###############################################################################
# {{{ close

cpdef close(np.ndarray dilation, int dist, res = None):

    if (dilation.dtype == 'bool'):
        dilation = dilation.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input images need to be binary (data type bool)')
    
    if (dilation.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _erode_2d(dilation, dist, res0, res1)
    elif (dilation.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _erode_3d(dilation, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
###############################################################################
# {{{ erode_map

cpdef erode_map(np.ndarray image, res = None):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')
    
    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _erode_map_2d(image, res0, res1)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _erode_map_3d(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "morphologyc.h":
    bint c_get_map_2d(
        unsigned short* image, 
        unsigned short* distance, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1, 
        int mode)


def _erode_map_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        double res0, 
        double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_get_map_2d(
        &image[0,0],
        &distance[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        1)

    assert status == 0
    return distance


cdef extern from "morphologyc.h":
    bint c_get_map_3d(
        unsigned short* image, 
        unsigned short* distance, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2, 
        int mode)


def _erode_map_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        double res0, 
        double res1, 
        double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_get_map_3d(
        &image[0,0,0],
        &distance[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        1)

    assert status == 0
    return distance

# }}}
###############################################################################
# {{{ dilate_map

cpdef dilate_map(np.ndarray image, res = None):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')
    
    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _dilate_map_2d(image, res0, res1)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _dilate_map_3d(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


def _dilate_map_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] image, 
        double res0, 
        double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_get_map_2d(
        &image[0,0],
        &distance[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        0,
    )

    assert status == 0
    return distance


def _dilate_map_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] image, 
        double res0, 
        double res1, 
        double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(
        image,dtype=np.uint16)
    
    status = c_get_map_3d(
        &image[0,0,0],
        &distance[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        0,
    )

    assert status == 0
    return distance

# }}}
###############################################################################
# {{{ open_map

cpdef open_map(np.ndarray erosion, res = None):

    if not (erosion.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (erosion.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _open_map_2d(erosion, res0, res1)
    elif (erosion.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _open_map_3d(erosion, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "morphologyc.h":
    bint c_open_map_2d(
        unsigned short* erosion, 
        unsigned short* opening, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1)


def _open_map_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] erosion, 
        double res0, 
        double res1):
    
    erosion = np.ascontiguousarray(erosion)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] opening = np.empty_like(
        erosion,dtype=np.uint16)
    
    status = c_open_map_2d(
        &erosion[0,0],
        &opening[0,0],
        erosion.shape[0],
        erosion.shape[1],
        res0,
        res1,
    )

    assert status == 0
    return opening


cdef extern from "morphologyc.h":
    bint c_open_map_3d(
        unsigned short* erosion, 
        unsigned short* opening, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2)


def _open_map_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion, 
        double res0, 
        double res1, 
        double res2):
    
    erosion = np.ascontiguousarray(erosion)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] opening = np.empty_like(
        erosion,dtype=np.uint16)
    
    status = c_open_map_3d(
        &erosion[0,0,0],
        &opening[0,0,0],
        erosion.shape[0],
        erosion.shape[1],
        erosion.shape[2],
        res0,
        res1,
        res2,
    )

    assert status == 0
    return opening

# }}}
###############################################################################
# {{{ close_map

cpdef close_map(np.ndarray dilation, res = None):

    if not (dilation.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (dilation.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return _close_map_2d(dilation, res0, res1)
    elif (dilation.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return _close_map_3d(dilation, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')


cdef extern from "morphologyc.h":
    bint c_close_map_2d(
        unsigned short* dilation, 
        unsigned short* closing, 
        int dim0, 
        int dim1, 
        double res0, 
        double res1)


def _close_map_2d(
        np.ndarray[np.uint16_t, ndim=2, mode="c"] dilation, 
        double res0, 
        double res1):
    
    dilation = np.ascontiguousarray(dilation)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] closing = np.empty_like(
        dilation,dtype=np.uint16)
    
    status = c_close_map_2d(
        &dilation[0,0],
        &closing[0,0],
        dilation.shape[0],
        dilation.shape[1],
        res0,
        res1,
    )

    assert status == 0
    return closing


cdef extern from "morphologyc.h":
    bint c_close_map_3d(
        unsigned short* dilation, 
        unsigned short* closing, 
        int dim0, 
        int dim1, 
        int dim2, 
        double res0, 
        double res1, 
        double res2)


def _close_map_3d(
        np.ndarray[np.uint16_t, ndim=3, mode="c"] dilation, 
        double res0, 
        double res1, 
        double res2):
    
    dilation = np.ascontiguousarray(dilation)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] closing = np.empty_like(
        dilation,dtype=np.uint16)
    
    status = c_close_map_3d(
        &dilation[0,0,0],
        &closing[0,0,0],
        dilation.shape[0],
        dilation.shape[1],
        dilation.shape[2],
        res0,
        res1,
        res2,
    )

    assert status == 0
    return closing

# }}}
###############################################################################
